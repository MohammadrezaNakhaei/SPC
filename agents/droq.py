# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
from copy import deepcopy
import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal 
from utils.layers import mlp, weight_init
from utils.distributions import SquashedNormal
from tensordict import TensorDict
from tensordict.nn import CudaGraphModule
from termcolor import colored
torch.set_float32_matmul_precision('high')

class DroQ(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.actor = mlp(cfg.obs_dim, [cfg.hidden_dim]*cfg.hidden_depth, 2*cfg.action_dim).to(cfg.device)
        self.q1 = mlp(cfg.obs_dim+cfg.action_dim, [cfg.hidden_dim]*cfg.hidden_depth, 1, dropout=cfg.dropout).to(cfg.device)
        self.q2 = mlp(cfg.obs_dim+cfg.action_dim, [cfg.hidden_dim]*cfg.hidden_depth, 1, dropout=cfg.dropout).to(cfg.device)
        self.q1_target = deepcopy(self.q1).requires_grad_(False)
        self.q2_target = deepcopy(self.q2).requires_grad_(False)
        capturable = cfg.cuda_graph and not cfg.compile
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr, betas=(0.9, 0.999), capturable=capturable)
        self.critic_optim = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), 
            lr=cfg.critic_lr, betas=(0.9, 0.999), capturable=capturable
            )
        if cfg.tune_alpha:
            self.log_alpha = torch.tensor(np.log(cfg.init_alpha), device=cfg.device, requires_grad=True)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr, betas=(0.9, 0.999), capturable=capturable)
            self.target_entropy = float(cfg.target_entropy)
        else:
            self.log_alpha = torch.tensor(np.log(cfg.alpha), device=cfg.device) 
        self.apply(weight_init)
        self.cfg = cfg
        if cfg.compile:
            print(colored('Compiling update and select_action function with torch.compile...'), 'red')
            mode = 'reduce-overhead' if cfg.compile and not cfg.cuda_graph else None
            self._update = torch.compile(self._update, mode=mode)
            self._select_det_action = torch.compile(self._select_det_action, mode=mode)
            self._select_sto_action = torch.compile(self._select_sto_action, mode=mode)

        if cfg.cuda_graph:
            print(colored('Using cudagraph for update and select_action function with...'), 'red')
            self._update = CudaGraphModule(self._update)
            self._select_sto_action = CudaGraphModule(self._select_sto_action)
            self._select_det_action = CudaGraphModule(self._select_det_action)

    def __repr__(self):
        repr = 'DroQ Agent\n'
        modules = ['Policy', 'Q1', 'Q2', 'log_alpha',]
        for i, m in enumerate([self.actor, self.q1, self.q2, self.log_alpha,]):
            repr += f"{modules[i]}: {m}\n"
        repr += "Learnable parameters: {:,}".format(self.total_params)
        return repr
        
    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def policy_output(self, obs):
        mu, logstd = self.actor(obs).chunk(2, dim=-1)
        logstd = torch.tanh(logstd)
        logstd =  self.cfg.log_std_min + 0.5 * (self.cfg.log_std_max - self.cfg.log_std_min) * (logstd +1)
        std = torch.exp(logstd)
        dist = SquashedNormal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        return action, dist.mean, log_prob
    
    def _update(self, obss, actions, rewards, next_obss, terminals ):
        # print(obss.shape, actions.shape, next_obss.shape, rewards.shape, terminals.shape)
        # update critic
        q1, q2 = self.q1(obss, actions), self.q2(obss, actions)
        with torch.no_grad():
            alpha = torch.exp(self.log_alpha)
            next_actions, _, next_log_probs = self.policy_output(next_obss)
            next_q = torch.min(self.q1_target(next_obss, next_actions), self.q2_target(next_obss, next_actions)) - alpha* next_log_probs
            target_q = rewards + self.cfg.discount * (1 - terminals) * next_q
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # update actor
        a, _, log_probs = self.policy_output(obss)
        q1a, q2a = self.q1(obss, a), self.q2(obss, a)

        actor_loss = -torch.min(q1a, q2a).mean() + alpha.detach() * log_probs.mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self.cfg.tune_alpha:
            with torch.no_grad():
                _, _, log_probs = self.policy_output(obss)
            alpha_loss = (-torch.exp(self.log_alpha) *(log_probs + self.target_entropy)).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

        self.sync()

        result = TensorDict({
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'average_q1': q1,
            'average_q2': q2,
            'entropy': -log_probs,
            
        })
        if self.cfg.tune_alpha:
            result.update({
                'alpha_loss': alpha_loss, 
                'alpha': alpha,
            })

        return result.mean().detach()
    
    @torch.no_grad()
    def select_action(self, obs, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.cfg.device)
        torch.compiler.cudagraph_mark_step_begin()
        action = self._select_det_action(obs) if deterministic else self._select_sto_action(obs)
        return action.cpu().numpy()
    
    @torch.no_grad()
    def _select_det_action(self, obs):
        _, mu, _ = self.policy_output(obs)
        return mu 
    
    @torch.no_grad()
    def _select_sto_action(self, obs):
        action, _, _ = self.policy_output(obs)
        return action
    
    @torch.no_grad()
    def sync(self,):
        soft_update_params(self.q1, self.q1_target, self.cfg.tau)
        soft_update_params(self.q2, self.q2_target, self.cfg.tau)

    def update(self, obss, actions, rewards, next_obss, terminals):
        torch.compiler.cudagraph_mark_step_begin()
        return self._update(obss, actions, rewards, next_obss, terminals)
    
    def save(self, fp):
        torch.save(self.state_dict(), fp)

    def load(self, fp):
        state_dict = torch.load(fp)
        self.load_state_dict(state_dict)

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +(1 - tau) * target_param.data)