import numpy as np 
from gymnasium.envs.mujoco.ant_v5 import AntEnv
from gymnasium.spaces import Box

class AntDirEnvCustom(AntEnv):
    def __init__(self, idx=0, **kwargs):
        super().__init__(**kwargs)
        self.num_tasks = 40
        self.num_train = 20
        self.num_eval_id = 10
        self.num_eval_ood = 10
        self._goal_idx = idx
        self.tasks = self.sample_tasks()
        self._task = self.tasks[self._goal_idx]
        self._goal = self._task['goal']
        self._max_episode_steps = 200
        self._step = 0
        self.observation_space = Box(-np.inf, np.inf, shape=(27,), dtype=np.float32)
        
    def _get_obs(self):
        # this is gym ant obs, should use rllab?
        # if position is needed, override this in subclasses
        return np.concatenate([
            self.data.qpos.flat[2:],
            self.data.qvel.flat,
        ])
    
    def step(self, action):
        torso_xyz_before = np.array(self.get_body_com("torso"))

        direct = (np.cos(self._goal), np.sin(self._goal))

        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = np.dot((torso_velocity[:2]/self.dt), direct)

        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.2 and state[2] <= 1.0
        terminated = not notdone
        ob = self._get_obs()
        self._step += 1
        truncated = self._step >= self._max_episode_steps
        return ob, reward, terminated, truncated, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            torso_velocity=torso_velocity,
        )

    def sample_tasks(self,):
        directions = np.linspace(0, 2*np.pi, self.num_tasks+1)[:-1]
        tasks = [{'goal': direction} for direction in directions]
        return tasks


    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal'] # assume parameterization of task by single vector
        self._goal_idx = idx
        self.reset()
        
    def get_all_task_idx(self):
        return range(self.num_tasks)
    
    def reset(self, *args, **kwargs):
        self._step = 0
        return super().reset(*args, **kwargs)
        
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
