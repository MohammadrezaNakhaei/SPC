import hydra
import numpy as np
import torch 
from gymnasium.vector import SyncVectorEnv
from termcolor import colored

from envs import make_env 
from utils.buffer import MultiReplayBuffer
from utils.seed import set_seed
from utils.logger import Logger
from agents.offline_metal_rl import OfflineMetaRL 
from tqdm import trange


@hydra.main(version_base='1.3', config_path="./cfgs", config_name="omrl")
def train(cfg):
    set_seed(cfg.seed)
    envs = make_env(env_id=cfg.env_name)
    cfg.agent.obs_dim = envs.observation_space.shape[0]
    cfg.agent.action_dim = envs.action_space.shape[0]
    envs.action_space.seed(cfg.seed)

    logger = Logger(cfg, print_log=False)
    print('='*50)
    agent = OfflineMetaRL(cfg.agent)
    policy = agent.agent
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
    # used for only storing samples for dataset
    print('='*50)
    task_modes = envs.task_modes()
    

    def evaluate(task_id, context, n_episodes, ):
        envs.reset_task(task_id)
        policy.infer_posterior(context)
        total_reward = 0
        counters =0
        obss, _ = envs.reset()
        while counters<n_episodes:
            actions = policy.select_action(obss[None, :], deterministic=True)[0]
            next_obss, rewards, terminated, truncated, _ = envs.step(actions)
            dones = terminated or truncated
            obss = next_obss
            total_reward += rewards
            if dones:
                obss = envs.reset()[0]
                counters += 1
        return total_reward/n_episodes
    
    def zero_shot(task_id, n_episodes):
        envs.reset_task(task_id)
        total_reward = 0
        for i in range(n_episodes):
            obss, _ = envs.reset()
            policy.clear_z()
            while True:
                actions = policy.select_action(obss[None, :], deterministic=True)[0]
                next_obss, rewards, terminated, truncated, _ = envs.step(actions)
                policy.update_context(obss, actions, rewards, next_obss)
                policy.infer_posterior()
                dones = terminated or truncated
                obss = next_obss
                total_reward += rewards
                if dones: break
        return total_reward/n_episodes       
        

    def _collect_context(task_id, n_steps, update_context, random_sample, resample_every, random_steps, ):
        envs.reset_task(task_id)
        policy.clear_z()
        policy.sample_z(use_ib=random_sample)
        obss, _ = envs.reset()
        for step in range(n_steps):
            if step<=random_steps:
                actions = envs.action_space.sample()
            else:
                actions = policy.select_action(obss[None, :], deterministic=False)[0]
            next_obss, rewards, terminated, truncated, _ = envs.step(actions)
            policy.update_context(obss, actions, rewards, next_obss)
            if update_context:
                policy.infer_posterior()
            if resample_every and step%resample_every==0:
                policy.sample_z(use_ib=random_sample)
            obss = next_obss
            dones = terminated or truncated
            if dones:
                    obss = envs.reset()[0]
        return policy.context 


    def collect_online_context(task_id, n_steps, random_sample=False, resample_every=None,):
        return _collect_context(
            task_id, n_steps=n_steps, update_context=False, 
            random_sample=random_sample, resample_every=resample_every, 
            random_steps=0)
    

    def collect_np_context(task_id, n_steps, random_steps=0, ):
        return _collect_context(
            task_id, n_steps=n_steps, update_context=True, 
            random_sample=False, resample_every=None, 
            random_steps=random_steps, )
    
    def collect_offline_context(task_id, n_steps):
        offline_context = buffer.sample([task_id,], n_steps, return_next_action=False)
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
            task_idx = np.random.choice(task_modes['train'], cfg.meta_batch_size).tolist()
            batch = buffer.sample(task_idx, cfg.batch_size, return_next_action=False)
            context = buffer.sample(task_idx, cfg.context_batch_size, return_next_action=False)
            train_info = agent.update(batch, context,task_idx, step=itr)
        step = cfg.iter_per_epoch * (epoch+1)
        train_info.update(dict(step=step))
        logger.log(train_info)
        eval_logs = dict(step=step)


        for mode in ['train', 'id', 'ood']:
            total_offline = 0
            total_online = 0
            total_nonprior = 0
            total_zeroshot = 0
            for i in task_modes[mode]:
            # collect contexts
                offline_context = collect_offline_context(i, cfg.num_context)
                online_context = collect_online_context(i, cfg.num_context, cfg.random_init_z, cfg.resample_every, )
                np_context = collect_np_context(i, cfg.num_context, cfg.random_np_steps,)
                # evalute
                eval_offline = evaluate(i, offline_context, cfg.num_eval_episodes)
                eval_online = evaluate(i, online_context, cfg.num_eval_episodes)
                eval_np = evaluate(i, np_context, cfg.num_eval_episodes)
                eval_zero = zero_shot(i, cfg.num_eval_episodes)
                total_offline += eval_offline 
                total_online += eval_online
                total_nonprior += eval_np 
                total_zeroshot += eval_zero
                if cfg.log_all:
                    eval_logs[f'{mode}/{i}_offline'] = eval_offline
                    eval_logs[f'{mode}/{i}_online'] = eval_online
                    eval_logs[f'{mode}/{i}_non_prior'] = eval_np
                    eval_logs[f'{mode}/{i}_zero_shot'] = eval_zero
            eval_logs[f'{mode}/avg_offline'] = total_offline / len(task_modes[mode])
            eval_logs[f'{mode}/avg_online'] = total_online / len(task_modes[mode])
            eval_logs[f'{mode}/avg_non_prior'] = total_nonprior / len(task_modes[mode])
            eval_logs[f'{mode}/avg_zero_shot'] = total_zeroshot / len(task_modes[mode])            
        logger.log(eval_logs, 'eval')
    print(colored('Saving agent', 'green', attrs=["bold"]))
    agent.save(f'{logger.log_dir}/agent.pth')
    print(colored('Training end', 'green', attrs=["bold"]))

if __name__ == '__main__':
    #add a change 
    train()