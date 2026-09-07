import numpy as np 
from gymnasium.envs.mujoco.half_cheetah_v5 import HalfCheetahEnv
from gymnasium.spaces import Box

class HalfCheetahVelEnvCustom(HalfCheetahEnv):
    def __init__(self, idx=0, **kwargs):
        super().__init__(**kwargs)
        self.num_tasks = 40
        self.num_train = 20
        self.num_eval_id = 10
        self.num_eval_ood = 10
        
        self._goal_idx = idx
        self.tasks = self.sample_tasks()
        self._task = self.tasks[self._goal_idx]
        self._goal_vel = self._task['velocity']
        self._goal = self._goal_vel
        self._max_episode_steps = 200
        self._step = 0
        self.observation_space = Box(-np.inf, np.inf, shape=(20,), dtype=np.float32)
        
    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat[1:],
            self.data.qvel.flat,
            self.get_body_com("torso").flat,
        ]).astype(np.float32).flatten()
    
    def step(self, action):
        xposbefore = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = -1.0 * abs(forward_vel - self._goal_vel)
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        self._step += 1
        terminatede = False
        truncated = True if self._step >= self._max_episode_steps else False
        infos = dict(reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost, task=self._task)
        return (observation, reward, terminatede, truncated, infos)

    def sample_tasks(self, ):
        velocities = np.linspace(0.1, 4.0, num=self.num_tasks)
        tasks = [{'velocity': velocity} for velocity in velocities]
        return tasks

    def get_all_task_idx(self):
        return range(self.num_tasks)
    
    def reset(self, *args, **kwargs):
        self._step = 0
        return super().reset(*args, **kwargs)


    def reset_task(self, idx):
        self._goal_idx = idx
        self._task = self.tasks[idx]
        self._goal_vel = self._task['velocity']
        self._goal = self._goal_vel
        self.reset()
        
    def get_task(self, ):
        return self._task
    
    def get_idx(self,):
        return self._goal_idx 
    
    def task_modes(self,):
        total = np.arange(5, 35)
        evals = np.arange(6, 35, 3)
        trains = np.setdiff1d(total, evals)
        ood = np.concatenate([np.arange(0, 5), np.arange(35, 40)])
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