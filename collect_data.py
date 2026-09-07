import hydra
import time

from envs import make_meta_env as make_env 
from utils.buffer import ReplayBuffer
from utils.seed import set_seed
from utils.logger import Logger

from agents.droq import DroQ
from termcolor import colored


@hydra.main(version_base='1.3', config_path="./cfgs", config_name="data_collection")
def train(cfg):
    set_seed(cfg.seed)
    env = make_env(cfg.env_name, seed=cfg.seed)
    eval_env = make_env(cfg.env_name, seed=cfg.seed)
    env.reset_task(cfg.goal_idx)
    eval_env.reset_task(cfg.goal_idx)
    env.action_space.seed(cfg.seed)
    cfg.agent.obs_dim = env.observation_space.shape[0]
    cfg.agent.action_dim = env.action_space.shape[0]
    logger = Logger(cfg)
    print('='*50)
    agent = DroQ(cfg.agent)
    print(agent)
    print('='*50)
    print(colored('Replay buffer used in training', 'green', attrs=["bold"]))
    buffer = ReplayBuffer(
        cfg.agent.obs_dim, cfg.agent.action_dim, 
        cfg.replay_buffer_capacity, cfg.device
        )
    # used for only storing samples for dataset
    print('='*50)
    print(colored('Replay buffer used to store data as dataset for OMRL', 'green', attrs=["bold"]))
    data_buffer = ReplayBuffer(
        cfg.agent.obs_dim, cfg.agent.action_dim, 
        cfg.dataset_size, 'cpu'        
        )
    print('='*50)
    
    obs, info, done = *env.reset(), False

    def eval(num_episode):
        # evalute the agent deterministicly
        total_rewards = 0
        total_success = 0 
        for i in range(num_episode):
            obs, info , done = *eval_env.reset(), False 
            if cfg.save_video and i==0:
                logger.video.init(eval_env, enabled=True)
            while not done:
                action = agent.select_action(obs, deterministic=True)
                obs, reward, term, trunc , info = eval_env.step(action)
                total_rewards += reward
                done = term or trunc
                if cfg.save_video and i==0:
                    logger.video.record(eval_env)
            total_success += info.get('success', 0)
        return total_rewards/num_episode, total_success/num_episode


    def eval_sample(num_episode):
        # evalute the agent stochastically and add samples to the buffer
        total_rewards = 0
        total_success = 0
        for i in range(num_episode):
            obs, info , done = *eval_env.reset(), False 
            while not done:
                action = agent.select_action(obs,)
                next_obs, reward, term, trunc , info = eval_env.step(action)
                total_rewards += reward
                data_buffer.add(obs, action, reward, next_obs, term)
                done = term or trunc
                obs = next_obs
            success = info.get('success', 0)
            total_success += success
        return total_rewards/num_episode, total_success/num_episode

    episode = 0
    episode_return = 0
    train_logs = {}
    start_time = time.time()
    for step in range(cfg.num_train_steps+1):
        if step and step%cfg.eval_frequency==0:
            # evaluate
            eval_returns, eval_success = eval(cfg.num_eval_episodes)
            eval_sample_returns, eval_sample_success = eval_sample(cfg.num_eval_sample_episodes)
            eval_logs = dict(
                episode_reward=eval_returns, traj_returns=eval_sample_returns, 
                episode_success=eval_success, traj_success=eval_sample_success)
            eval_logs.update(dict(step=step, episode=episode, total_time=time.time()-start_time))
            logger.log(eval_logs, category='eval')
            if cfg.save_video:
                logger.video.save(step)

        if done:
            episode += 1
            episode_success = info.get('success', 0)
            train_logs.update(dict(
                episode_success=episode_success, episode_reward=episode_return, 
                step=step, episode=episode, total_time=time.time()-start_time))
            logger.log(train_logs)
            episode_return = 0
            obs, info, done = *env.reset(), False
            
        if step>cfg.num_seed_steps:
            data = buffer.sample(cfg.batch_size)
            train_logs = agent.update(*data)
            train_logs = train_logs.cpu().numpy()
            action = agent.select_action(obs)
        else:
            action = env.action_space.sample()
        
        next_obs, reward, term, trunc, info = env.step(action)
        done = term or trunc     
        buffer.add(obs, action, reward, next_obs, term)
        episode_return += reward
        obs = next_obs
    logger.finish()
    print(colored('Saving Dataset', 'green', attrs=["bold"]))
    data_buffer.save(f'{logger.log_dir}/dataset.pth')
    if cfg.save_agent:
        print(colored('Saving agent', 'green', attrs=["bold"]))
        agent.save(f'{logger.log_dir}/agent.pth')
    print(colored('Training end', 'green', attrs=["bold"]))


if __name__ == '__main__':
    #add a change 
    train()