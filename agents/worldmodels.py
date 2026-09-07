#!/usr/bin/env python3
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from utils.layers import mlp, SimNorm
from utils.fsq import FSQ
from utils.ensemble import Ensemble



class ContinuousMSE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg.latent_dim 
        self.z_dim = cfg.z_dim
        self.encoder =  mlp(cfg.obs_dim, [cfg.enc_hidden_dim]*cfg.enc_hidden_depth, self.latent_dim)
        if cfg.use_tar_enc:
            self.encoder_tar = copy.deepcopy(self.encoder).requires_grad_(False)

        trans_out_dim = self.latent_dim
        self._trans = mlp(
            self.latent_dim + cfg.action_dim + cfg.z_dim, 
            [cfg.model_hidden_dim]*cfg.model_hidden_depth, 
            trans_out_dim)
        rewards = [mlp(self.latent_dim + cfg.action_dim + cfg.z_dim, 
            [cfg.model_hidden_dim]*cfg.model_hidden_depth, 
            1) for _ in range(cfg.num_rewards)]
        self._reward = Ensemble(rewards)
        self.rho = torch.tensor([self.cfg.rho**t for t in range(self.cfg.horizon)], device=self.cfg.device)

    def encode(self, obs, target: bool = False):
        """
        Encode observation into latent space. Should return latent representation and index in discrete case
        """
        latent = self.encoder_tar(obs) if target else self.encoder(obs)
        return latent, None
    
    def trans(self, latent, a, task_z,):
        """ 
        Predict next latent representation using action and task information. Return next latent representation and logits.
        """
        # Returns logits for each class
        delta = self._trans(latent, a, task_z)
        return latent + delta, None
    
    def reward(self, latent, a, task_z, mode='avg', penalty_coef=0):
        """Predicts reward with optiional penalty for uncertainty"""
        r = self._reward(latent, a, task_z)
        if mode=='avg':
            return r.mean(0)
        if mode=='min':
            return r.min(0)[0]
        if mode=='std_penalty':
            assert self.cfg.num_rewards>1
            return r.mean(0)-penalty_coef*r.std(0)
        else:
            raise NotImplementedError

    def loss(self, obss, actions, rewards, next_obss, dones, task_z, allow_backward_z=True):
        ### dimensions: seq_length, batch_size, dim ###
        if not allow_backward_z:
            task_z = task_z.detach()
        seq_len, batch_size, _ = obss.shape
        # temporal consistency and reward loss
        tc_loss = torch.zeros(1, device=self.cfg.device)
        reward_loss = torch.zeros(1, device=self.cfg.device)

        #Create targets
        with torch.no_grad():
            latent_tar, indices_tar = self.encode(next_obss, target=True)

        # Latent rollout
        latents =  torch.empty(
                self.cfg.horizon + 1,
                batch_size,
                # self.cfg.latent_dim,
                self.latent_dim,
                device=self.cfg.device,)
            
        latent, _ = self.encode(obss[0])
        latents[0] = latent
        
        for t in range(self.cfg.horizon):
            # Predict next latent
            next_latent, _ = self.trans(latent=latent, a=actions[t], task_z=task_z)
            latents[t+1] = next_latent
            # Don't forget this
            latent = next_latent

        reward_loss = self.rewards_loss(latents, actions, task_z, rewards, dones)
        tc_loss = self.tc_loss(latents, latent_tar)
        return tc_loss, reward_loss 
    
    def rewards_loss(self, latents, actions, task_z, rewards, dones):
        seq_len, batch_size, _ = actions.shape
        r_pred = self._reward(latents[:-1], actions, task_z.repeat(seq_len, 1, 1))[..., 0]
        rewards = rewards.squeeze(-1)
        rewards = rewards.broadcast_to(r_pred.shape)
        dones = dones.squeeze(-1)
        # assert r_pred.ndim == 2 and rewards.ndim == 2
        # _reward_loss = (r_pred - rewards) ** 2
        _reward_loss = torch.mean((r_pred - rewards) ** 2, 0) # now it is seq_batch_size
        _rho_reward_loss = self.rho * torch.mean(
            # (1 - terminateds_or_dones) * _reward_loss, -1
            (1 - dones) * _reward_loss,
            -1,
        )
        reward_loss = torch.mean(_rho_reward_loss)
        return reward_loss
    
    def tc_loss(self, latents, tar_latents, ):
        _tc_loss = torch.mean((latents[1:]-tar_latents)**2, -1)
        _rho_tc_loss = self.rho * torch.mean(_tc_loss, -1)
        tc_loss = torch.mean(_rho_tc_loss)
        return tc_loss

    def soft_update_params(self):
        # Update the tar network
        if self.cfg.use_tar_enc:
            for params, params_target in zip(self.encoder.parameters(), self.encoder_tar.parameters()):
                params_target.data.lerp_(params.data, self.cfg.tau)
    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    
    
class ContinuousCosine(ContinuousMSE):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def tc_loss(self, latents, tar_latents):
        _tc_loss = -F.cosine_similarity(latents[1:], tar_latents, dim=-1)
        _rho_tc_loss = self.rho * torch.mean(_tc_loss, -1)
        tc_loss = torch.mean(_rho_tc_loss)
        return tc_loss



class SimnormMSE(ContinuousMSE):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.sim_norm = SimNorm(cfg.simnorm_dim)
        
    def encode(self, obs, target = False):
        latent = self.encoder_tar(obs) if target else self.encoder(obs)
        latent = self.sim_norm(latent)
        return latent, None



class SimnormCosine(ContinuousMSE):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.sim_norm = SimNorm(cfg.simnorm_dim)
        
    def encode(self, obs, target = False):
        latent = self.encoder_tar(obs) if target else self.encoder(obs)
        latent = self.sim_norm(latent)
        return latent, None
    
    def tc_loss(self, latents, tar_latents):
        _tc_loss = -F.cosine_similarity(latents[1:], tar_latents, dim=-1)
        _rho_tc_loss = self.rho * torch.mean(_tc_loss, -1)
        tc_loss = torch.mean(_rho_tc_loss)
        return tc_loss



class DiscreteMSE(ContinuousMSE):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.num_channels = len(cfg.fsq_levels)
        self._fsq = FSQ(levels=cfg.fsq_levels)

    def encode(self, obs, target: bool = False):
        latent = self.encoder_tar(obs) if target else self.encoder(obs)
        return self._fsq(latent)



class DiscreteCosine(DiscreteMSE):
    def __init__(self, cfg):
        super().__init__(cfg)

    def tc_loss(self, latents, tar_latents):
        _tc_loss = -F.cosine_similarity(latents[1:], tar_latents, dim=-1)
        _rho_tc_loss = self.rho * torch.mean(_tc_loss, -1)
        tc_loss = torch.mean(_rho_tc_loss)
        return tc_loss



class DiscreteCE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        ##### Configure FSQ stuff #####
        self.org_latent_dim = copy.copy(cfg.latent_dim)
        self.num_channels = len(cfg.fsq_levels)
        self._fsq = FSQ(levels=cfg.fsq_levels)

        self.latent_dim = cfg.latent_dim * self.num_channels
        self.z_dim = cfg.z_dim
        self.encoder =  mlp(cfg.obs_dim, [cfg.enc_hidden_dim]*cfg.enc_hidden_depth, self.latent_dim)

        ##### Init encoder #####
        if cfg.use_tar_enc:
            self.encoder_tar = copy.deepcopy(self.encoder).requires_grad_(False)

        trans_out_dim = int(self.org_latent_dim * self._fsq._fsq.codebook_size)
        self._trans = mlp(
            self.latent_dim + cfg.action_dim + cfg.z_dim, 
            [cfg.model_hidden_dim]*cfg.model_hidden_depth, 
            trans_out_dim)
        rewards = [mlp(self.latent_dim + cfg.action_dim + cfg.z_dim, 
            [cfg.model_hidden_dim]*cfg.model_hidden_depth, 
            1) for _ in range(cfg.num_rewards)]
        self._reward = Ensemble(rewards)
        
    def encode(self, obs, target: bool = False):
        latent = self.encoder_tar(obs) if target else self.encoder(obs)
        return self._fsq(latent)


    def trans(self, latent, a, task_z, unc_prop_mode:str = None):
        # Returns logits for each class
        logits = self._trans(latent, a, task_z)
        logits = logits.reshape(
            -1,
            self.org_latent_dim,
            self._fsq._fsq.codebook_size,
        )

        if unc_prop_mode is None:
            unc_prop_mode = self.cfg.unc_prop_mode

        # Convert latent state logits to an actual latent state
        if unc_prop_mode in ["mode", "max"]:
            # TODO this is the same as the mode
            # Note this has no gradients so should only be used for MPC
            indices = torch.max(logits, -1)[1]
            next_z = self._fsq._fsq.implicit_codebook[
                indices.to(torch.long)
            ].flatten(-2)

        elif "sample-no-grad" in unc_prop_mode:
            indices = torch.vmap(torch.multinomial, randomness="different")(
                torch.softmax(logits, -1), num_samples=1
            )[..., 0]
            next_z = self._fsq._fsq.implicit_codebook[indices].flatten(-2)

        elif "sample" in unc_prop_mode:
            z_one_hot = torch.nn.functional.gumbel_softmax(
                logits, tau=1, hard=True, dim=-1
            )
            codebook = self._fsq._fsq.implicit_codebook
            next_z = einsum(z_one_hot, codebook, "b d c, c l -> b d l")
            next_z = rearrange(next_z, "b d l -> b (d l)")

        elif "weighted-avg" in unc_prop_mode:
            probs = F.softmax(logits, dim=-1)
            codebook = self._fsq._fsq.implicit_codebook
            next_z = einsum(probs, codebook, "b d c, c l -> b d l")
            next_z = rearrange(next_z, "b d l -> b (d l)")

        return next_z, logits 
    
    def reward(self, latent, a, task_z, mode='avg', penalty_coef=0):
        r = self._reward(latent, a, task_z)
        if mode=='avg':
            return r.mean(0)
        if mode=='min':
            return r.min(0)[0]
        if mode=='std_penalty':
            assert self.cfg.num_rewards>1
            return r.mean(0)-penalty_coef*r.std(0)
        else:
            raise NotImplementedError

    def loss(self, obss, actions, rewards, next_obss, dones, task_z, allow_backward_z=True):
        ### dimensions: seq_length, batch_size, dim ###
        if not allow_backward_z:
            task_z = task_z.detach()
        seq_len, batch_size, _ = obss.shape
        # temporal consistency and reward loss
        tc_loss = torch.zeros(1, device=self.cfg.device)
        reward_loss = torch.zeros(1, device=self.cfg.device)

        #Create targets
        with torch.no_grad():
            latent_tar, indices_tar = self.encode(next_obss, target=True)

        # Latent rollout
        codes =  torch.empty(
                self.cfg.horizon + 1,
                batch_size,
                # self.cfg.latent_dim,
                self.latent_dim,
                device=self.cfg.device,)
        
        logits = torch.empty(
                self.cfg.horizon + 1,
                batch_size,
                self.org_latent_dim,
                # int(self.cfg.latent_dim / self.num_channels),
                self._fsq._fsq.codebook_size,
                device=self.cfg.device,)

            
        latent, _ = self.encode(obss[0])
        codes[0] = latent

        for t in range(self.cfg.horizon):
            # Predict next latent
            next_code, next_logits = self.trans(latent=latent, a=actions[t], task_z=task_z)
            codes[t+1] = next_code
            logits[t+1] = next_logits
            # Don't forget this
            latent = next_code

        rho = torch.tensor([self.cfg.rho**t for t in range(self.cfg.horizon)], device=self.cfg.device)
        
        # terminateds_or_dones = terminateds_or_dones.to(torch.int)

        #####Reward prediction loss #####
        r_pred = self._reward(codes[:-1], actions, task_z.repeat(seq_len, 1, 1))[..., 0]
        rewards = rewards.squeeze(-1)
        rewards = rewards.broadcast_to(r_pred.shape)
        dones = dones.squeeze(-1)
        # assert r_pred.ndim == 2 and rewards.ndim == 2
        # _reward_loss = (r_pred - rewards) ** 2
        _reward_loss = torch.mean((r_pred - rewards) ** 2, 0) # now it is seq_batch_size
        _rho_reward_loss = rho * torch.mean(
            # (1 - terminateds_or_dones) * _reward_loss, -1
            (1 - dones) * _reward_loss,
            -1,
        )
        reward_loss = torch.mean(_rho_reward_loss)
        _tc_loss = torch.vmap(torch.vmap(F.cross_entropy))(
                logits[1:], indices_tar.to(torch.long))


        _rho_tc_loss = rho * torch.mean((1 - dones) * _tc_loss, -1)
        tc_loss = torch.mean(_rho_tc_loss)
        return tc_loss, reward_loss 

    def soft_update_params(self):
        # Update the tar network
        if self.cfg.use_tar_enc:
            for params, params_target in zip(self.encoder.parameters(), self.encoder_tar.parameters()):

                params_target.data.lerp_(params.data, self.cfg.tau)
    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

