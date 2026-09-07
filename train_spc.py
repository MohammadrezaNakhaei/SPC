import hydra
import numpy as np
import torch 
from torch.utils.data import DataLoader
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from termcolor import colored

from envs import make_meta_env 
from utils.buffer import MultiReplayBuffer, MultiSequenceBuffer
from utils.seed import set_seed
from utils.logger import Logger
from agents.spc import CWM
from tqdm import trange


@hydra.main(version_base='1.3', config_path="./cfgs", config_name="spc")
def train(cfg):
    set_seed(cfg.seed)
    envs = SyncVectorEnv([lambda i=i:make_meta_env(env_id=cfg.env_name, idx=i, seed=cfg.seed) for i in range(cfg.num_tasks)])
    # set the task for each env, it seems that the initiation fails to work
    for i, env in enumerate(envs.envs):
        env.reset_task(i)
    cfg.agent.obs_dim = envs.single_observation_space.shape[0]
    cfg.agent.action_dim = envs.single_action_space.shape[0]
    envs.action_space.seed(cfg.seed)
    task_modes = envs.envs[0].task_modes()
    cfg.agent.num_train_tasks = len(task_modes['train'])
    logger = Logger(cfg, print_log=False)
    print('='*50)
    agent = CWM(cfg.agent)
    policy = agent
    print(agent)
    print('='*50)
    print(colored('Multi Task Replay buffer used in training', 'green', attrs=["bold"]))
    buffer = MultiReplayBuffer(
        buffer_idxs=range(cfg.num_tasks), 
        obs_dim=cfg.agent.obs_dim, 
        action_dim=cfg.agent.action_dim, 
        capacity=cfg.dataset_size,
        device=cfg.device,
    )
    print(colored('Loading datasets', 'yellow', attrs=["bold"]))
    for i in range(cfg.num_tasks):
        path = f'data/{cfg.env_name}/goal_idx{i}/dataset.pth'
        buffer.load(i, path)

    seq_buffer = MultiSequenceBuffer(buffer,
                                     seq_len = cfg.seq_len, 
                                     ep_length = cfg.max_episode_steps, 
                                    )
    # used for only storing samples for dataset
    print('='*50)
    
    def evaluate(context, n_episodes, ):
        policy.infer_posterior(context)
        total_reward = np.zeros(envs.num_envs)
        total_success = np.zeros(envs.num_envs)
        t0 = True
        counters = np.zeros(envs.num_envs, dtype=np.int8)
        obss, _ = envs.reset()
        while np.any(counters<n_episodes):
            actions = policy.select_action(obss, deterministic=True, t0=t0)
            t0=False
            next_obss, rewards, terminated, truncated, infos = envs.step(actions)
            dones = np.logical_or(terminated, truncated)
            obss = next_obss
            total_reward[counters<n_episodes] += rewards[counters<n_episodes]
            for ind, done in enumerate(dones):
                if done:
                    obss[ind] = envs.envs[ind].reset()[0]
                    counters[ind] += 1
                    t0=True
                    if 'success' in infos:
                        total_success[ind] += infos['success'][ind]
        return total_reward/n_episodes, total_success/n_episodes

    def zero_shot(n_episodes):
        total_reward = np.zeros(envs.num_envs)
        total_success = np.zeros(envs.num_envs)
        for i in range(n_episodes):
            obss, _ = envs.reset()
            policy.clear_z(envs.num_envs)
            notdones = np.ones(envs.num_envs, dtype=np.bool_)
            t0 = True
            while np.any(notdones):
                actions = policy.select_action(obss, deterministic=True, t0=t0)
                t0=False
                next_obss, rewards, terminated, truncated, infos = envs.step(actions)
                policy.update_context(obss, actions, rewards, next_obss)
                policy.infer_posterior()
                dones = np.logical_or(terminated, truncated)
                obss = next_obss
                total_reward[notdones] += rewards[notdones]
                notdones = np.logical_and(notdones, ~dones)
                for ind, done in enumerate(dones):
                    if done:
                        if 'success' in infos:
                            total_success[ind] += infos['success'][ind]
        return total_reward/n_episodes, total_success/n_episodes
        

    def _collect_context(n_steps, update_context, random_sample, resample_every, random_steps):
        policy.clear_z(envs.num_envs)
        policy.sample_z()
        obss, _ = envs.reset()
        t0=True
        for step in range(n_steps):
            if step<=random_steps:
                actions = envs.action_space.sample()
            else:
                actions = policy.select_action(obss, deterministic=False, t0=t0)
            t0=False
            next_obss, rewards, terminated, truncated, _ = envs.step(actions)
            policy.update_context(obss, actions, rewards, next_obss)
            if update_context:
                policy.infer_posterior()
            if resample_every and step%resample_every==0:
                policy.sample_z()
            obss = next_obss
            dones = np.logical_or(terminated, truncated)
            for ind, done in enumerate(dones):
                if done:
                    obss[ind] = envs.envs[ind].reset()[0]
                    t0=True
        return policy.context 


    def collect_online_context(n_steps, random_sample=False, resample_every=None,):
        return _collect_context(
            n_steps=n_steps, update_context=False, 
            random_sample=random_sample, resample_every=resample_every, 
            random_steps=0)
    

    def collect_np_context(n_steps, random_steps=0, ):
        return _collect_context(
            n_steps=n_steps, update_context=True, 
            random_sample=False, resample_every=None, 
            random_steps=random_steps, )
    
    def collect_offline_context(n_steps):
        offline_context = buffer.sample(range(cfg.num_tasks), n_steps, return_next_action=False)
        cont_obss, cont_actions, cont_rewards, cont_next_obses, cont_dones = offline_context 
        if agent.cfg.use_next_obs_in_context:
            offline_context = (cont_obss, cont_actions, cont_rewards, cont_next_obses)
        else:
            offline_context = (cont_obss, cont_actions, cont_rewards)
        offline_context = torch.cat(offline_context, dim=-1)
        return offline_context

    for epoch in range(cfg.epochs):
        # train the agent
        for itr in trange(cfg.iter_per_epoch, desc=f'Epoch: {epoch} Training the agent', miniters=25):
            train_idx = np.random.randint(0, len(task_modes['train']), cfg.meta_batch_size)
            task_idx = task_modes['train'][train_idx]
            batch = seq_buffer.sample(task_idx, cfg.batch_size)
            context = buffer.sample(task_idx, cfg.context_batch_size, return_next_action=False)
            train_info = agent.update(batch, context, train_idx, step=itr+1)
        step =  cfg.iter_per_epoch*(epoch+1)
        train_info.update(dict(step=step))
        logger.log(train_info)
        # collect contexts
        offline_context = collect_offline_context(cfg.num_context)
        online_context = collect_online_context(cfg.num_context, cfg.random_init_z, cfg.resample_every, )
        np_context = collect_np_context(cfg.num_context, cfg.random_np_steps,)
        # evalute
        eval_offline, offline_success = evaluate(offline_context, cfg.num_eval_episodes)
        eval_online, online_success = evaluate(online_context, cfg.num_eval_episodes)
        eval_np, np_success = evaluate(np_context, cfg.num_eval_episodes)
        eval_zero, zero_success = zero_shot(cfg.num_eval_episodes)
        eval_logs = dict(step=step)
        contexts = ['offline', 'online', 'non_prior', 'zero_shot']
        contexts = ['offline', 'online', 'non_prior', 'zero_shot']
        evals = [eval_offline, eval_online, eval_np, eval_zero]
        success_s = [offline_success, online_success, np_success, zero_success]
        for context, eval, success in zip(contexts, evals, success_s):
            for mode in ['train', 'id', 'ood']:
                eval_logs[f'{mode}/avg_{context}'] = eval[task_modes[mode]].mean()
                eval_logs[f'{mode}/avg_{context}_success'] = success[task_modes[mode]].mean()
                # log the results for all tasks
                if cfg.log_all: 
                    for ind in task_modes[mode]:
                        eval_logs[f'{mode}/{ind}_{context}'] = eval[ind]
        logger.log(eval_logs, 'eval')
    print(colored('Saving agent', 'green', attrs=["bold"]))
    agent.save(f'{logger.log_dir}/agent.pth')
    print(colored('Training end', 'green', attrs=["bold"]))

if __name__ == '__main__':
    #add a change 
    train()