#!/usr/bin/env python3
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Adam
from torch.distributions import Normal
from tensordict import TensorDict
from utils.layers import mlp, mlp_relu
from utils.fsq import FSQ
from utils.ensemble import Ensemble
from .worldmodels import ContinuousMSE, ContinuousCosine, SimnormMSE, SimnormCosine, DiscreteMSE, DiscreteCosine, DiscreteCE

worldmodels = dict(
    continuous_mse=ContinuousMSE,
    continuous_cosine=ContinuousCosine,
    simnorm_mse=SimnormMSE,
    simnorm_cosine=SimnormCosine,
    discrete_mse=DiscreteMSE,
    discrete_cosine=DiscreteCosine,
    discrete_ce=DiscreteCE
)


class CWM(nn.Module):
    def __init__(self, cfg, ):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.model = worldmodels[cfg.world_model](cfg).to(cfg.device)
        self.model_optimizer = AdamW(self.model.parameters(), cfg.model_lr)
        
        latent_dim = self.model.latent_dim 
        self.act_dim = cfg.action_dim
        self.z_dim = cfg.z_dim        
        self.actor = mlp(
            in_dim = latent_dim +self.z_dim, 
            mlp_dims=[cfg.actor_hidden_dim]*cfg.actor_hidden_depth, 
            out_dim=2*cfg.action_dim).to(cfg.device)
        qs = [
            mlp(
                in_dim=latent_dim + cfg.action_dim + self.z_dim, 
                mlp_dims=[cfg.q_hidden_dim]*cfg.q_hidden_depth, 
                out_dim=1).to(cfg.device)
            for _ in range(cfg.num_critics)]
        self.qs = Ensemble(qs)
        self.qs_target = copy.deepcopy(self.qs).requires_grad_(False)
        self.v = mlp(
            in_dim=latent_dim + self.z_dim, 
            mlp_dims=[cfg.q_hidden_dim]*cfg.q_hidden_depth, 
            out_dim=1).to(cfg.device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.qs_optimzier = Adam(self.qs.parameters(), lr=cfg.critic_lr)
        self.v_optimizer = Adam(self.v.parameters(), lr=cfg.critic_lr)

        context_encoder_input_dim = 2 * cfg.obs_dim + cfg.action_dim + 1 if cfg.use_next_obs_in_context else cfg.obs_dim + cfg.action_dim + 1
        self.context_encoder_input_dim = context_encoder_input_dim
        self.context_encoder = mlp(
                    context_encoder_input_dim, 
                    [cfg.context_hidden_dims]*cfg.context_hidden_depth, 
                    self.z_dim).to(cfg.device)
        self._fsq_context = FSQ(levels=cfg.fsq_context_levels).to(cfg.device)
        self.context_enc_opimizer = Adam(self.context_encoder.parameters(), lr=cfg.context_enc_lr)
        if cfg.use_infonce:
            assert not self.cfg.use_focal, 'Cannot use both focal and infonce'
            self.latent_history_means = torch.zeros(self.cfg.num_train_tasks, cfg.z_dim, device=cfg.device)

    def __repr__(self):
        repr = f'Contextual DCWM Agent\n'
        modules = ['Context Encoder', 'World Model', 'Policy', 'Q-Functions', 'Value Function']
        for i, m in enumerate([self.context_encoder, self.model, self.actor, self.qs, self.v]):
            repr += f"{modules[i]}: {m}\n"
        repr += "Learnable parameters: {:,} M\n".format(self.total_params/1e6)
        return repr
    
    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def clear_z(self, num_tasks=1):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        mu = torch.zeros((num_tasks, self.z_dim), device=self.cfg.device)
        var =  torch.ones((num_tasks, self.z_dim), device=self.cfg.device)
        self.z_means = mu
        self.z_vars = var
        self.sample_z()
        self.context = None

    def update_context(self, obs, action, reward, next_obs):
        if isinstance(reward, float): 
            reward = [reward]
        else:
            reward = reward[:, None] 
        if self.cfg.use_next_obs_in_context:
            new_context = np.concatenate([obs, action, reward, next_obs],axis=-1)
        else:
            new_context = np.concatenate([obs, action, reward], axis=-1)
        # add meta-batch if single transitions from one task
        if new_context.ndim==1:
            new_context = new_context[None, ...] 
        # add batch dimension to the context
        new_context = torch.as_tensor(new_context, dtype=torch.float32, device=self.cfg.device).unsqueeze(1)
        if self.context is None:
            self.context = new_context
        else:
            self.context = torch.cat([self.context, new_context], dim=1)

    def infer_posterior(self, context=None,):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        if context is None:
            context = self.context
        params = self.context_encoder(context)
        if self.cfg.use_context_fsq:
            params, _ = self._fsq_context(params)
        if self.cfg.use_tanh:
            params = torch.tanh(params)
        if self.cfg.use_l2_context_norm:
            # params = F.normalize(params, p=2, dim=-1)
            with torch.no_grad():
                norm = torch.norm(params, dim=-1, p=2, keepdim=True)
            params = params/norm
        self.z_means = torch.mean(params, dim=1) # dim: task, batch, feature (latent dim)
        if self.cfg.use_context_fsq_mean:
            self.z_means, _ = self._fsq_context(self.z_means)
        self.z_vars = torch.std(params, dim=1)
        self.sample_z()

    def sample_z(self,):
        self.z = self.z_means
    
    def sample_init_z(self, num_tasks, deterministic=False):
        if deterministic: 
            return torch.zeros((num_tasks, self.z_dim), device=self.cfg.device)
        return torch.randn(size=(num_tasks, self.z_dim), device=self.cfg.device)
    
    def get_task_z(self, context, batch_size):
        self.infer_posterior(context)
        self.sample_z()
        task_z = self.z
        # self.meta_batch * self.batch_size * dim(obs)
        task_z = [z.repeat(batch_size, 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)
        return task_z

    def encode_all(self, context):
        params = self.context_encoder(context)
        if self.cfg.use_context_fsq:
            params, _ = self._fsq_context(params)
        if self.cfg.use_tanh:
            params = torch.tanh(params)
        return params

    def update(self, batch, context, indices, step=0):
        """
        batch: task_batch, batch_size, seq_len, dim
        context: task_batch, context_size, dim
        step: int
        indices: task_batch
        """
        torch.compiler.cudagraph_mark_step_begin()
        indices = torch.tensor(indices, device=self.cfg.device, dtype=torch.long)
        self.context_enc_opimizer.zero_grad()
        result = TensorDict()
        cont_obss, cont_actions, cont_rewards, cont_next_obses, cont_dones = context 
        if self.cfg.use_next_obs_in_context:
            context = (cont_obss, cont_actions, cont_rewards, cont_next_obses)
        else:
            context = (cont_obss, cont_actions, cont_rewards)
        context = torch.cat(context, dim=-1)
        # task_batch, batch_size, seq_len, dim
        t, s, b, _ = batch[0].shape
        # reshape to seq_len, task_batch*batch_size, dim
        obss, actions, rewards, next_obss, dones = map(lambda x: x.swapaxes(0, 1).reshape(s, t*b, -1), batch)

        task_z = self.get_task_z(context, b)
        task_z = task_z.reshape(t*b, -1)
        # update the contextual world model less frequently
        if step%self.cfg.model_update_freq==0:
            result.update(self._update_model(obss, actions, rewards, next_obss, dones, task_z))
            self.model.soft_update_params()
            # update the context encoder
            result.update(self._update_encoder(task_z, indices, b))
        # expand task_z to have the same dimensions 
        task_z = task_z.detach().expand(s, t*b, -1)
        # flatten the tensors for updating actor and critic
        obss, actions, rewards, next_obss, dones, task_z = map(lambda x: x.reshape(t*b*s, -1), [obss, actions, rewards, next_obss, dones, task_z])
        with torch.no_grad():
            latents = self.model.encode(obss, target=False)[0]
            next_latents = self.model.encode(next_obss, target=False)[0]
        result.update(self._update_actor_critic(latents, actions, rewards, next_latents, dones, task_z))
        self.sync()
        return result
    
    def _update_model(self, obss, actions, rewards, next_obss, dones, task_z, ):
        self.model_optimizer.zero_grad()
        tc_loss, reward_loss = self.model.loss(obss, actions, rewards, next_obss, dones, task_z, allow_backward_z=self.cfg.allow_z_model)
        loss = self.cfg.consistency_coef * tc_loss + self.cfg.reward_coef * reward_loss
        # retain the computational graph and gradients if using contextual world model to train context enc
        loss.backward(retain_graph=self.cfg.allow_z_model) 
        self.model_optimizer.step()
        return TensorDict({
            'tc_loss': tc_loss,
            'reward_loss': reward_loss, 
            'total_loss': loss,
        }).mean().detach()
    
    def _update_actor_critic(self, latents, actions, rewards, next_latents, terminals, task_z):
        task_z = task_z.detach()
        qs = self.qs(latents, actions, task_z)
        v = self.v(latents, task_z)
        with torch.no_grad():
            q = torch.min(self.qs_target(latents, actions, task_z), dim=0)[0]
            exp_a = torch.exp((q - v) * self.cfg.temperature)
            exp_a = torch.clip(exp_a, None, 100.0)
        critic_v_loss = self._expectile_regression(q-v).mean()
        self.v_optimizer.zero_grad()
        critic_v_loss.backward()
        self.v_optimizer.step()

        with torch.no_grad():
            next_v = self.v(next_latents, task_z)
            target_q = rewards + self.cfg.discount * (1 - terminals) * next_v
            target_q = target_q.broadcast_to(qs.shape)
        
        critic_loss = F.mse_loss(qs, target_q)

        self.qs_optimzier.zero_grad()
        critic_loss.backward()
        self.qs_optimzier.step()

        dist = self.policy_output(latents, task_z.detach())
        policy_loss = -(exp_a * dist.log_prob(actions).sum(-1, keepdim=True)).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        result = TensorDict({
            'critic_loss': critic_loss,
            'average_q': q, 
            'v_loss': critic_v_loss,
            'policy_loss': policy_loss,
        })
        return result.mean().detach()
    
    def _update_encoder(self, task_z, indices, batch_size):
        loss = torch.zeros(1, device=self.cfg.device, requires_grad=True)
        if self.cfg.use_focal:
            focal_loss = self._focal_loss(indices, task_z, batch_size)
            loss = loss + self.cfg.focal_weight*focal_loss
        if self.cfg.use_infonce:
            means = task_z.reshape(-1, batch_size, self.cfg.z_dim)[:,0]
            infonce_loss = self._infonce_loss(means, indices)
            loss = loss + self.cfg.infonce_weight*infonce_loss
        loss.backward()
        self.context_enc_opimizer.step()
        result = TensorDict({})
        if self.cfg.use_focal: result.update({'focal_loss': focal_loss})
        if self.cfg.use_infonce: result.update({'infonce_loss': infonce_loss})
        return result.mean().detach()
    
    def _focal_loss(self, indices, task_z, batch_size):
        pos_z_loss = 0.
        neg_z_loss = 0.
        pos_cnt = 0
        neg_cnt = 0
        for i in range(len(indices)):
            idx_i = i * batch_size # index in task * batch dim
            for j in range(i+1, len(indices)):
                idx_j = j * batch_size # index in task * batch dim
                if indices[i] == indices[j]:
                    pos_z_loss += torch.sqrt(torch.mean((task_z[idx_i] - task_z[idx_j]) ** 2) + 1e-3)
                    pos_cnt += 1
                else:
                    neg_z_loss += 1/(torch.mean((task_z[idx_i] - task_z[idx_j]) ** 2) + 1e-3 * 100)
                    neg_cnt += 1
        return pos_z_loss/(pos_cnt + 1e-3) +  neg_z_loss/(neg_cnt + 1e-3)

    def _infonce_loss(self, means, indices):
        # list of index for each task
        task_cnt = self.latent_history_means.shape[0]
        mapping = {}
        for j, ind in enumerate(indices):
            if ind not in mapping:
                mapping[ind] = []
            mapping[ind].append(j)
        current_means = self.latent_history_means.clone()
        for ind, j_index in mapping.items():
            current_means[ind] = means[j_index].mean(0)
        # update history mean
        with torch.no_grad():
            self.latent_history_means = (1-self.cfg.infonce_tau) * self.latent_history_means + self.cfg.infonce_tau * current_means
        # compute distance matrix
        queries = current_means.unsqueeze(0).repeat(task_cnt, 1, 1)
        keys = self.latent_history_means.unsqueeze(1).repeat(1, task_cnt, 1).detach()
        distance_matrix = (torch.sum(torch.pow(queries - keys, 2), dim = -1) + 1e-6).sqrt()
        l_pos = torch.diag(distance_matrix).view(-1, 1)
        l_neg = distance_matrix
        logits = torch.cat([l_pos, l_neg], dim = 1)
        labels = torch.zeros(l_pos.shape[0], dtype = torch.long)
        labels = labels.to(distance_matrix.device)
        loss_fn = nn.CrossEntropyLoss()
        infonce_loss = loss_fn(- logits / self.cfg.infonce_radius, labels)
        return infonce_loss

    def _expectile_regression(self, diff):
        weight = torch.where(diff > 0, self.cfg.expectile, (1 - self.cfg.expectile))
        return weight * (diff**2)

    def policy_output(self, latent, task_z):
        mean_logstd = torch.tanh(self.actor(latent, task_z))
        mu, logstd = torch.chunk(mean_logstd, chunks=2, dim=-1)
        logstd =  self.cfg.log_std_min + 0.5 * (self.cfg.log_std_max - self.cfg.log_std_min) * (logstd +1)
        std = torch.exp(logstd)
        dist = Normal(mu, std)
        return dist
    
    @torch.no_grad()
    def select_action(self, obs, t0=False, deterministic=False):
        ''' 
        sample action from the policy, conditioned on the task embedding 
        '''
        z = self.z
        obs = torch.as_tensor(obs, device=self.cfg.device, dtype=torch.float32)
        latent = self.model.encode(obs)[0]
        if self.cfg.mpc:
            action, mppi_std = self.plan(latent, z, t0=t0, deterministic=deterministic)
        else:
            dist = self.policy_output(latent, z)
            action = dist.mean if deterministic else dist.sample()
        return torch.clamp(action, -1, 1).cpu().numpy()

    @torch.no_grad()
    def _pi(self, latent, task_z, deterministic=False):
        dist = self.policy_output(latent, task_z)
        action = dist.mean if deterministic else dist.sample()
        return torch.clamp(action, -1, 1)

    @torch.no_grad()
    def plan(self, latent, task_z, t0: bool = False, deterministic=False):
        """
        Plan a sequence of actions using the learned world model.

        Args:
            z (torch.Tensor): Latent state from which to plan.
            t0 (bool): Whether this is the first observation in the episode.
            eval_mode (bool): Whether to use the mean of the action distribution.

        Returns:
            torch.Tensor: Action to take in the environment.
        """
        batch_size = latent.shape[0]
        pi_actions = torch.empty(
            batch_size,
            self.cfg.plan_horizon,
            self.cfg.num_pi_trajs,
            self.act_dim,
            device=self.device,
        )
        actions = torch.empty(
            batch_size,
            self.cfg.plan_horizon,
            self.cfg.num_samples,
            self.act_dim,
            device=self.device,
        )
        mean = torch.zeros(
            batch_size, self.cfg.plan_horizon, self.act_dim, device=self.device
        )

        def single_mppi(latent, task_z, actions, pi_actions, mean, prev_mean):
            # Sample policy trajectories
            
            if self.cfg.num_pi_trajs > 0:
                _latent = latent.expand(self.cfg.num_pi_trajs, *latent.shape)
                _task_z = task_z.expand(self.cfg.num_pi_trajs, *task_z.shape)
                for t in range(self.cfg.plan_horizon - 1):
                    pi_actions[t] = self._pi(_latent, _task_z, deterministic=False,)
                    _latent, _ = self.model.trans(
                        latent=_latent, a=pi_actions[t], task_z=_task_z,
                        unc_prop_mode=self.cfg.plan_unc_prop_mode,
                    )
                pi_actions[-1] = self._pi(_latent, _task_z, deterministic=False,)

            # Initialize state and parameters
            latent = latent.expand(self.cfg.num_samples, *latent.shape)
            task_z = task_z.expand(self.cfg.num_samples, *task_z.shape)
            std = self.cfg.max_std * torch.ones(
                self.cfg.plan_horizon, self.act_dim, device=self.device)
            
            if not t0:
                mean[:-1] = prev_mean[1:]
            if self.cfg.num_pi_trajs > 0:
                actions[:, : self.cfg.num_pi_trajs] = pi_actions

            # Iterate MPPI
            for _ in range(self.cfg.iterations):
                # Sample actions
                actions[:, self.cfg.num_pi_trajs :] = (
                    mean.unsqueeze(1)
                    + std.unsqueeze(1)
                    * torch.randn(
                        self.cfg.plan_horizon,
                        self.cfg.num_samples - self.cfg.num_pi_trajs,
                        self.act_dim,
                        device=std.device,
                    )
                ).clamp(-1, 1)

                # Compute elite actions
                value = self._single_estimate_value(latent, task_z, actions).nan_to_num_(0)
                elite_idxs = torch.topk(
                    value.squeeze(1), self.cfg.num_elites, dim=0
                ).indices
                elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

                # Update parameters
                max_value = elite_value.max(0)[0]
                score = torch.exp(self.cfg.plan_temperature * (elite_value - max_value))
                score /= score.sum(0)
                mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (
                    score.sum(0) + 1e-9
                )
                std = torch.sqrt(
                    torch.sum(
                        score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2,
                        dim=1,
                    )
                    / (score.sum(0) + 1e-9)
                )
                std = std.clamp(self.cfg.min_std, self.cfg.max_std)
                # std.clamp_(self.cfg.min_std, self.cfg.max_std)

            act_dist = torch.distributions.Categorical(score[:, 0])
            act_idx = act_dist.sample()
            # actions = elite_actions[:, act_idx]
            actions = torch.index_select(elite_actions, 1, act_idx)[:, 0, :]
            a, std = actions[0], std[0]
            if not deterministic:
                a += std * torch.randn(self.act_dim, device=std.device)
            return a, mean, std

        if hasattr(self, "_prev_mean") and not t0:
            prev_mean = self._prev_mean
        else:
            prev_mean = torch.zeros(
                batch_size,
                self.cfg.plan_horizon,
                self.act_dim,
                device=self.device,
            )
        # single_mppi(z_td[0], actions[0], pi_actions[0], mean[0], prev_mean[0])
        a, new_prev_mean, std = torch.vmap(
            single_mppi, in_dims=(0, 0, 0, 0, 0, 0), randomness="different"
        )(latent, task_z, actions, pi_actions, mean, prev_mean)

        if self.cfg.mppi_use_mean:
            a = new_prev_mean[:, 0]

        self._prev_mean = new_prev_mean
        a = torch.clamp(a, -1 ,1)
        return a, std

    @torch.no_grad()
    def _single_estimate_value(self, latent, task_z, actions):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        for t in range(self.cfg.plan_horizon):
            reward = self.model.reward(latent=latent, a=actions[t], task_z=task_z, mode=self.cfg.reward_mode, penalty_coef=self.cfg.reward_penalty)
            # reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
            latent, _ = self.model.trans(
                latent=latent, a=actions[t], task_z=task_z, unc_prop_mode=self.cfg.plan_unc_prop_mode
            )
            G += discount * reward
            discount *= self.cfg.rho
        if self.cfg.plan_with_value:
            G += discount * self.v(latent, task_z)
        return G 
    
    def sync(self):
        for params, params_target in zip(self.qs.parameters(), self.qs_target.parameters()):
            params_target.data.lerp_(params.data, self.cfg.tau)

    def save(self, fp):
        torch.save(self.state_dict(), fp)

    def load(self, fp):
        state_dict = torch.load(fp)
        self.load_state_dict(state_dict)
        