import numpy as np
from gymnasium import Env, spaces


class PointRobot(Env):

    def __init__(self, idx=0, **kwargs):
        self.num_tasks = 40
        self.num_train = 20
        self.num_moderate = 10
        self.num_extreme = 10
        self._goal_idx = idx
        self.tasks = self.sample_tasks()
        self._task = self.tasks[self._goal_idx]
        self._goal = self._task['goal']

        self._max_episode_steps = 40
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)
        self._step = 0

    def step(self, action):
        self._state = self._state + action
        # before exiting the hallway
        if self._state[1]<0:
            reward = -1+self._state[1]
            self._state[0] = np.clip(self._state[1], -0.1, 0.1) # imaginary wall 
            self._state[1] = max(self._state[1], -1.5) # imaginary wall
        else:
            reward = -np.linalg.norm(self._state-self._goal, )
        done = False
        ob = self._get_obs()
        self._step += 1
        if self._step >= self._max_episode_steps:
            done = True
        return ob, reward, False, done, dict()

    def sample_tasks(self,):
        angles = np.linspace(0, np.pi, self.num_tasks)
        tasks = [{'goal': np.array([np.cos(angle), np.sin(angle)])} for angle in angles]
        return tasks

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal'] # assume parameterization of task by single vector
        self._goal_idx = idx
        self.reset()
        
    def get_all_task_idx(self):
        return range(self.num_tasks)
    
    def reset(self, *args, **kwargs):
        # reset to a random location on the unit square
        self._state = np.array([0.0, -1.0])
        self._step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.copy(self._state)
        
    def get_task(self, ):
        return self._task
    
    def get_idx(self,):
        return self._goal_idx 
    
    def task_modes(self,):
        return {
            'train': np.array(list(range(10,30))),
            'id': np.array(list(range(5, 10)) + list(range(30,35))),
            'ood': np.array(list(range(0,5)) + list(range(35,40)))
        }
        
    def get_mode(self, ):
        idx = self._goal_idx
        for k,v in self.task_modes().items():
            if idx in v:
                return k