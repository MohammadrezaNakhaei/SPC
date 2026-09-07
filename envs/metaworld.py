import gymnasium as gym 
import metaworld 
import numpy as np

def _make_ml1(env_name, ):
    ml1 = metaworld.ML1(env_name, seed=42)
    env = ml1.train_classes[env_name]() 
    tasks = ml1.train_tasks 
    return env, tasks


class MetaWorldEnv():
    def __init__(self, env_name, idx=0, seed=0, action_repeat=2, **kwargs):
        self.num_tasks = 50
        self.num_train = 40
        self.num_eval_id = 10
        self.num_eval_ood = 0
        self._max_episode_steps = 200
        self._seed = seed
        self.action_repeat = action_repeat
        self.env_name = env_name
        self._env, self._tasks = _make_ml1(self.env_name,)
        self._env.seed(self._seed)
        self.reset_task(idx)

    def reset_task(self, idx):
        self._goal_idx = idx
        self._env.set_task(self._tasks[idx])
        
    def get_all_task_idx(self):
        return range(self.num_tasks)
        
    def get_task(self, ):
        return self._task
    
    def get_idx(self,):
        return self._goal_idx 
    
    def task_modes(self,):        
        trains = np.arange(0, 40)
        evals = np.arange(40, 50)
        ood = []
        return {
            'train': trains,
            'id': evals,
            'ood': ood,
        }
        
    def get_mode(self, ):
        idx = self._goal_idx
        for k,v in self.task_modes().items():
            if idx in v:
                return k

    def reset(self, *args, **kwargs):
        self._step = 0
        return self._env.reset()
    
    def step(self, action):
        total_reward = 0
        for _ in range(self.action_repeat):
            obs, reward, terminated, truncated, info = self._env.step(action)
            total_reward += reward
            if terminated or truncated:
                break 
        self._step += 1
        if self._step >= self._max_episode_steps:
            truncated = True
        return obs, total_reward, terminated, truncated, info

    def __getattr__(self, name):
        return getattr(self._env, name)
    
    