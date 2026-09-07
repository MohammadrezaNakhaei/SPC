import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F
from utils.layers import mlp_relu, mlp
from torch.distributions import Normal
from tensordict.nn import CudaGraphModule
from termcolor import colored


def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


class ContextualAgent(nn.Module):
    def __init__(self, cfg):
        '''
        create context encoder and policy
        '''
        super().__init__()
        self.cfg = cfg 
        context_encoder_input_dim = 2 * cfg.obs_dim + cfg.action_dim + 1 if cfg.use_next_obs_in_context else cfg.obs_dim + cfg.action_dim + 1
        context_encoder_output_dim = cfg.latent_dim * 2 if cfg.use_information_bottleneck else cfg.latent_dim
        self.context_encoder_input_dim = context_encoder_input_dim
        self.context_output_dim = context_encoder_output_dim
        
        model_class = mlp if cfg.use_layernorm else mlp_relu
        self.context_encoder = model_class(
            context_encoder_input_dim, 
            [cfg.enc_hidden_dim]*cfg.enc_hidden_depth, 
            context_encoder_output_dim,).to(cfg.device)

        self.policy = model_class(cfg.obs_dim+cfg.latent_dim, [cfg.actor_hidden_dim]*cfg.actor_hidden_depth, 2*cfg.action_dim).to(cfg.device)
        self.use_ib = cfg.use_information_bottleneck
        self.use_next_obs_in_context = cfg.use_next_obs_in_context
        self.latent_dim = cfg.latent_dim
        self.clear_z()

        if cfg.compile: self._compile()
        if cfg.cuda_graph: self._cuda_graph()
    
    def _compile(self,):
        torch.set_float32_matmul_precision('high')
        print(colored('Using torch.compile for acceleration', 'red'))
        mode = 'reduce-overhead' if not self.cfg.cuda_graph else None 
        self._policy_output = torch.compile(self._policy_output, mode=mode)

    def _cuda_graph(self,):
        print(colored('Using cuda graph for acceleration', 'red'))
        self._policy_output = CudaGraphModule(self._policy_output)
        
        # self._update_actor_critic = CudaGraphModule(self._update_actor_critic)
    def clear_z(self, num_tasks=1):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        mu = torch.zeros((num_tasks, self.latent_dim), device=self.cfg.device)
        var =  torch.ones((num_tasks, self.latent_dim), device=self.cfg.device)
        self.z_means = mu
        self.z_vars = var
        self.sample_z()
        self.context = None

    def detach_z(self):
        self.z = self.z.detach()


    def update_context(self, obs, action, reward, next_obs):
        if isinstance(reward, float): 
            reward = [reward]
        else:
            reward = reward[:, None] 
        if self.use_next_obs_in_context:
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


    def compute_kl_div(self):
        prior = torch.distributions.Normal(
            torch.zeros((self.latent_dim), device=self.cfg.device), 
            torch.ones((self.latent_dim), device=self.cfg.device)
            )
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    def infer_posterior(self, context=None):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        if context is None:
            context = self.context
        params = self.context_encoder(context)
        if self.cfg.use_tanh:
            params = torch.tanh(params)
        # with probabilistic z, predict mean and variance of q(z | c)
        if self.use_ib:
            mu = params[..., :self.latent_dim]
            sigma_squared = F.softplus(params[..., self.latent_dim:])
            # permutation invariant encoding
            z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
            self.z_means = torch.stack([p[0] for p in z_params])
            self.z_vars = torch.stack([p[1] for p in z_params])
        # sum rather than product of gaussians structure
        else:
            self.z_means = torch.mean(params, dim=1) 
            self.z_vars = torch.std(params, dim=1)
        self.sample_z()


    def sample_z(self, use_ib=None):
        if use_ib is None:
            use_ib = self.use_ib
        if use_ib:
            posteriors = torch.distributions.Normal(self.z_means, self.z_vars)
            self.z = posteriors.rsample() # reparameterization trick
        else:
            self.z = self.z_means
    
    def sample_init_z(self, num_tasks, deterministic=False):
        if deterministic: 
            return torch.zeros((num_tasks, self.latent_dim), device=self.cfg.device)
        return torch.randn(size=(num_tasks, self.latent_dim), device=self.cfg.device)

    @torch.no_grad()
    def select_action(self, obs, deterministic=False):
        ''' 
        sample action from the policy, conditioned on the task embedding 
        '''
        z = self.z
        obs = torch.as_tensor(obs, device=self.cfg.device, dtype=torch.float32)

        dist = self._policy_output(obs, z)
        action = dist.mean if deterministic else dist.sample()
        return torch.clamp(action, -1, 1).cpu().numpy()
    
    @torch.no_grad()
    def _policy_output(self, obs, z):
        return self.policy_output(obs, z)
    
    def policy_output(self, obs, task_z):
        mean_logstd = torch.tanh(self.policy(obs, task_z))
        mu, logstd = torch.chunk(mean_logstd, chunks=2, dim=-1)
        logstd =  self.cfg.log_std_min + 0.5 * (self.cfg.log_std_max - self.cfg.log_std_min) * (logstd +1)
        std = torch.exp(logstd)
        dist = Normal(mu, std)
        return dist

    def get_task_z(self, context, batch_size):
        self.infer_posterior(context)
        self.sample_z()
        task_z = self.z
        # self.meta_batch * self.batch_size * dim(obs)
        task_z = [z.repeat(batch_size, 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)
        return task_z

    def encode_all(self, context):
        if self.use_ib:
            mu_logstd = self.context_encoder(context)
            mu, logstd = torch.chunk(mu_logstd, chunks=2, dim=-1)
            std = F.softplus(logstd)
            # reparameterization trick
            posteriors = torch.distributions.Normal(mu, std)
            z = posteriors.rsample() 
        else:
            z =  self.context_encoder(context)
        if self.cfg.use_tanh:   
            z = torch.tanh(z)
        return z

    @property
    def networks(self):
        return [self.context_encoder, self.policy]


class RNNEncoder(nn.Module):
    def __init__(self, context_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(context_dim, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, )
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, context, hidden_state=None):
        x = self.fc1(context)
        x = F.mish(x)
        x, h = self.rnn(x, hidden_state)
        x = self.fc2(x)
        return x, h
    
    
class RNNContextAgent(ContextualAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.context_encoder = RNNEncoder(
            self.context_encoder_input_dim, 
            cfg.enc_hidden_dim, 
            self.context_output_dim
        ).to(cfg.device)
        
    def clear_z(self, num_tasks=1):
        super().clear_z(num_tasks)
        self._rnn_hidden = None
        self.new_context = None
        
    def update_context(self, obs, action, reward, next_obs):
        super().update_context(obs, action, reward, next_obs)
        self.new_context = self.context[:, -1, :]  # last context element
            
    def infer_posterior(self, context=None):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        if context is None:
            params, self._rnn_hidden = self.context_encoder(self.new_context[None], self._rnn_hidden) # add time dimension
            params = params[-1]  # take last time step
        else:
            assert context.ndim == 4, 'Context shape: (Time, Task Batch, Mini Batch, Features)'
            # assert context.shape[1] == self.z_means.shape[0], 'Context batch size must match number of tasks when resetted'
            s, t, b, _ = context.shape
            context = context.reshape(s, t*b, -1)
            params, h = self.context_encoder(context)
            params = params[-1].reshape(t, b, -1).mean(1)
        if self.cfg.use_tanh:
            params = torch.tanh(params)
        # with probabilistic z, predict mean and variance of q(z | c)
        if self.use_ib:
            raise NotImplementedError

        # sum rather than product of gaussians structure
        else:
            self.z_means = params # dim: task, batch, feature (latent dim)
        self.sample_z()
        
    def encode_all(self, context):
        if self.use_ib:
            raise NotImplementedError
        s, t, b, _ = context.shape
        context = context.reshape(s, t*b, -1)
        z, h = self.context_encoder(context)
        z = z[-1].reshape(t, b, -1)
        if self.cfg.use_tanh:   
            z = torch.tanh(z)
        return z
        