from copy import deepcopy
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from .context_policy import ContextualAgent, RNNContextAgent
from utils.layers import mlp_relu, mlp, soft_update_params
from utils.ensemble import Ensemble
from tensordict import TensorDict
from tensordict.nn import CudaGraphModule
from termcolor import colored


class OfflineMetaRL(nn.Module):
    def __init__(self, cfg):
        '''
        FOCAL, CSRO, UNICRON agent based on Implicit Q-Learning
        '''
        super().__init__()
        # initilize policy and context encoder
        self.agent = ContextualAgent(cfg) if not cfg.use_rnn_encoder else RNNContextAgent(cfg)
        self.actor = self.agent.policy 
        self.context_encoder = self.agent.context_encoder
        self.cfg = cfg

        # offline rl networks
        model_type = mlp if cfg.use_layernorm else mlp_relu
        self.q1 = model_type(cfg.obs_dim+cfg.action_dim+cfg.latent_dim, [cfg.q_hidden_dim]*cfg.q_hidden_depth, 1, dropout=cfg.dropout,).to(cfg.device)
        self.q2 = model_type(cfg.obs_dim+cfg.action_dim+cfg.latent_dim, [cfg.q_hidden_dim]*cfg.q_hidden_depth, 1, dropout=cfg.dropout,).to(cfg.device)
        self.v = model_type(cfg.obs_dim+cfg.latent_dim, [cfg.q_hidden_dim]*cfg.q_hidden_depth, 1, dropout=cfg.dropout,).to(cfg.device) 
        self.q1_target = deepcopy(self.q1).requires_grad_(False)
        self.q2_target = deepcopy(self.q2).requires_grad_(False)

        capturable = cfg.cuda_graph and not cfg.compile
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr,  capturable=capturable)
        self.critic_optim = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.critic_lr, capturable=capturable
        )
        self.v_optim = optim.Adam(self.v.parameters(), cfg.critic_lr, capturable=capturable)
        self._expectile = cfg.expectile

        # CLUB (MI upper bound) in CSRO
        if cfg.use_club:
            club_input_dim = cfg.obs_dim + cfg.action_dim
            if cfg.use_next_obs_in_context: club_input_dim +=1
            self.club = model_type(club_input_dim, [cfg.enc_hidden_dim]*cfg.enc_hidden_depth, 2*cfg.latent_dim, act=nn.Tanh()).to(cfg.device)
            self.club_optim = optim.Adam(self.club.parameters(), lr=cfg.club_lr, capturable=capturable)
            self.club_input_dim = club_input_dim
        # Decoder to predict next state and reward UNICORN
        if cfg.use_decoder:
            self.decoder =  model_type(cfg.obs_dim+cfg.action_dim+cfg.latent_dim, [cfg.decoder_hidden_dim]*cfg.decoder_hidden_depth, cfg.obs_dim+1).to(cfg.device)
            self.encoder_optim = optim.Adam(list(self.context_encoder.parameters()) + list(self.decoder.parameters()), lr=cfg.encoder_lr, capturable=capturable)
        else:
            self.encoder_optim = optim.Adam(self.context_encoder.parameters(), lr=cfg.encoder_lr, capturable=capturable)
        # infoNCE contrastive loss
        if cfg.use_infonce:
            assert not self.cfg.use_focal, 'Cannot use both focal and infonce'
            self.latent_history_means = torch.zeros(self.cfg.num_train_tasks, cfg.latent_dim, device=cfg.device)
        # Acceleration  based compile and cuda grah
        if cfg.compile: self._compile()
        if cfg.cuda_graph: self._cuda_graph()
    
    def _compile(self,):
        torch.set_float32_matmul_precision('high')
        print(colored('Using torch.compile for acceleration', 'red'))
        mode = 'reduce-overhead' if not self.cfg.cuda_graph else None 
        self._update_club = torch.compile(self._update_club, mode=mode)        
        # self._update_encoder = torch.compile(self._update_encoder, mode=mode)
        self._update_actor_critic = torch.compile(self._update_actor_critic, mode=mode)

    def _cuda_graph(self,):
        print(colored('Using cuda graph for acceleration', 'red'))
        self._update_club = CudaGraphModule(self._update_club)
        # self._update_encoder = CudaGraphModule(self._update_encoder)
        self._update_actor_critic = CudaGraphModule(self._update_actor_critic)
        
    def __repr__(self):
        repr = f'{self.cfg.name} Agent\n'
        modules = ['Context Encoder', 'Policy', 'Q-Function 1', 'Q-Function 2', 'Value Function']
        for i, m in enumerate([self.context_encoder, self.actor, self.q1, self.q2, self.v]):
            repr += f"{modules[i]}: {m}\n"
        if self.cfg.use_club:
            repr += f'CLUB: {self.club}\n'
        if self.cfg.use_decoder:
            repr += f'Decoder: {self.decoder}\n'
        repr += "Learnable parameters: {:,} M\n".format(self.total_params/1e6)
        return repr
    
    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def update(self, batch, context, indices, step=0):
        indices = torch.tensor(indices, device=self.cfg.device, dtype=torch.long)
        torch.compiler.cudagraph_mark_step_begin()
        result = TensorDict({})
        # obss, actions, rewards, next_obss, dones, next_actions = batch 
        cont_obss, cont_actions, cont_rewards, cont_next_obses, cont_dones = context 
        if self.cfg.use_next_obs_in_context:
            context = (cont_obss, cont_actions, cont_rewards, cont_next_obses)
        else:
            context = (cont_obss, cont_actions, cont_rewards)
        context = torch.cat(context, dim=-1)
        if self.cfg.use_club:
            result.update(self._update_club(context))
        # reshape the tensors to 2d
        result.update(self._update_encoder(batch, context, indices))
        t, b, _ = batch[0].shape
        obss, actions, rewards, next_obss, dones = map(lambda x: x.reshape(t*b, -1), batch)
        with torch.no_grad():
            task_z = self.agent.get_task_z(context, b)
            # task_z = task_z.reshape(t*b, -1)
        # update the critic
        result.update(self._update_actor_critic(obss, actions, rewards, next_obss, dones, task_z))
        self.sync()
        return result
    

    def _update_actor_critic(self, obss, actions, rewards, next_obss, terminals, task_z):
        v = self.v(obss, task_z)
        with torch.no_grad():
            q1_tar, q2_tar = self.q1_target(obss, actions, task_z), self.q2_target(obss, actions,task_z)
            q = torch.min(q1_tar, q2_tar)
            exp_a = torch.exp((q - v) * self.cfg.temperature)
            exp_a = torch.clip(exp_a, None, 100.0)
        critic_v_loss = self._expectile_regression(q-v).mean()
        self.v_optim.zero_grad()
        critic_v_loss.backward()
        self.v_optim.step()

        with torch.no_grad():
            next_v = self.v(next_obss, task_z)
            target_q = rewards + self.cfg.discount * (1 - terminals) * next_v
        q1, q2 = self.q1(obss, actions, task_z), self.q2(obss, actions,task_z)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        dist = self.agent.policy_output(obss, task_z)
        policy_loss = -(exp_a * dist.log_prob(actions).sum(-1, keepdim=True)).mean()
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()
        
        result = TensorDict({
            'critic_loss': critic_loss,
            'average_q1': q1, 
            'average_q2': q2,
            'v_loss': critic_v_loss,
            'policy_loss': policy_loss,
        })
        return result.mean().detach()
    
    def _update_encoder(self, batch, context, indices, ):
        self.encoder_optim.zero_grad()
        loss = torch.zeros(1, dtype=torch.float32, device=self.cfg.device).requires_grad_()
        z = self.agent.encode_all(context)
        t, b, _ = z.shape
        if self.cfg.use_club:
            with torch.no_grad():
                z_param = self.club(context[..., :self.club_input_dim]).detach()
                z_mean, z_log_var  = z_param.chunk(2, dim=-1)
                z_var = F.softplus(z_log_var)
            position = - ((z-z_mean)**2/z_var).mean()
            z_mean_expand = z_mean[:, :, None, :].expand(-1, -1, b, -1).reshape(t, b**2, -1)
            z_var_expand = z_var[:, :, None, :].expand(-1, -1, b, -1).reshape(t, b**2, -1)
            z_target_repeat = z.repeat(1, b, 1)
            negative = - ((z_target_repeat-z_mean_expand)**2/z_var_expand).mean()
            club_loss = (position - negative) #self.club_loss_weight * 
            loss = loss + self.cfg.club_loss_weight*club_loss
        
        means = torch.mean(z, dim=1)

        if self.cfg.use_focal:
            focal_loss = self._focal_loss(means, indices)
            loss = loss + self.cfg.focal_loss_weight*focal_loss

        if self.cfg.use_infonce:
            infonce_loss = self._infonce_loss(means, indices)
            loss = loss + self.cfg.infonce_weight*infonce_loss

        if self.cfg.use_decoder:
            obss, actions, rewards, next_obss, dones = batch
            predictions = self.decoder(obss, actions, means.unsqueeze(1).repeat(1, b, 1))
            target = torch.cat([next_obss-obss, rewards], dim=-1)
            deocder_loss = F.mse_loss(predictions, target)
            loss = loss + self.cfg.decoder_weight * deocder_loss
            
        if self.cfg.use_information_bottleneck:
            self.agent.infer_posterior(context)
            kl_div = self.agent.compute_kl_div()
            loss = loss + self.cfg.kl_weight * kl_div
        
        if self.cfg.use_l2_reg:
            assert not self.cfg.use_information_bottleneck
            reg_loss = F.mse_loss(means, torch.zeros_like(means))
            loss = loss + self.cfg.l2_reg_weight * reg_loss

        loss.backward()    
        self.encoder_optim.step()
        result = TensorDict({'encoder_loss': loss})
        if self.cfg.use_club: result.update({'club_encoder_loss': club_loss})
        if self.cfg.use_focal: result.update({'focal_loss': focal_loss})
        if self.cfg.use_decoder: result.update({'decoder_loss': deocder_loss})
        if self.cfg.use_information_bottleneck: result.update({'kl_div': kl_div})
        if self.cfg.use_l2_reg: result.update({'l2_reg_loss': reg_loss})
        if self.cfg.use_infonce: result.update({'infonce_loss': infonce_loss})
        return result.mean().detach()

    def _update_club(self, context, ):
        self.club_optim.zero_grad()
        with torch.no_grad():
            z_target = self.agent.encode_all(context) 
        z_param = self.club(context[..., :self.club_input_dim])
        z_mean, z_log_var  = z_param.chunk(2, dim=-1)
        z_var = F.softplus(z_log_var)
        club_model_loss = ((z_target- z_mean)**2/(2*z_var) + torch.log(torch.sqrt(z_var))).mean()
        club_model_loss.backward()
        self.club_optim.step()
        return TensorDict({
            'club_loss': club_model_loss,
        }).mean().detach()
            
    def _focal_loss(self, means, indices):
        pos_z_loss = 0.
        neg_z_loss = 0.
        pos_cnt = 0
        neg_cnt = 0
        for i in range(len(indices)):
            idx_i = i # * batch_size # index in task * batch dim
            for j in range(i+1, len(indices)):
                idx_j = j # * batch_size # index in task * batch dim
                if indices[i] == indices[j]:
                    pos_z_loss += torch.sqrt(torch.mean((means[idx_i] - means[idx_j]) ** 2) + 1e-3)
                    pos_cnt += 1
                else:
                    neg_z_loss += 1/(torch.mean((means[idx_i] - means[idx_j]) ** 2) + 1e-3 * 100)
                    neg_cnt += 1
        return pos_z_loss/(pos_cnt + 1e-3) +  neg_z_loss/(neg_cnt + 1e-3)

    def _infonce_loss(self, means, indices):
        # list of index for each task
        task_cnt = self.latent_history_means.shape[0]
        mapping = {}
        for j,ind in enumerate(indices):
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

    def sync(self):
        soft_update_params(self.q1, self.q1_target, self.cfg.tau)
        soft_update_params(self.q2, self.q2_target, self.cfg.tau)

    def _expectile_regression(self, diff):
        weight = torch.where(diff > 0, self._expectile, (1 - self._expectile))
        return weight * (diff**2)

    def save(self, fp):
        torch.save(self.state_dict(), fp)

    def load(self, fp):
        state_dict = torch.load(fp)
        self.load_state_dict(state_dict)