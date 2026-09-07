import numpy as np 
from gymnasium.envs.mujoco.ant_v5 import AntEnv
from gymnasium.spaces import Box 

class AntGoalEnvCustom(AntEnv):
    def __init__(self, idx=0, **kwargs):
        super().__init__(**kwargs)
        self.num_tasks = 40
        self.num_train = 20
        self.num_moderate = 10
        self.num_extreme = 10
        
        self._goal_idx = idx
        self.tasks = self.sample_tasks()
        self._task = self.tasks[self._goal_idx]
        self._goal = self._task['goal']
        self._max_episode_steps = 200
        self._step = 0
        self.observation_space = Box(-np.inf, np.inf, shape=(29,), dtype=np.float32)
        
    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))

        goal_reward = -np.sum(np.abs(xposafter[:2] - self._goal)) # make it happy, not suicidal

        ctrl_cost = .1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        terminated = False
        self._step += 1
        truncated = True if self._step >= self._max_episode_steps else False
        ob = self._get_obs()
        return ob, reward, terminated, truncated, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
        )

    def sample_tasks(self):
        # training tasks
        angle = np.linspace(-np.pi/2, np.pi/2, 10)
        radius = [0.8, 1.2]
        training_goals = [(r * np.cos(a), r * np.sin(a)) for r in radius for a in angle]
        # moderate goals, same direction further away
        id_goals = [(1.0 * np.cos(a), 1.0 * np.sin(a)) for a in angle]
        ood_goals = [(1.6 * np.cos(a), 1.6 * np.sin(a)) for a in angle]
        goals = training_goals + id_goals + ood_goals
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.flat,
            # np.clip(self.data.cfrc_ext, -1, 1).flat,
        ])
        
    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal'] # assume parameterization of task by single vector
        self._goal_idx = idx
        self.reset()
        
    def get_all_task_idx(self):
        return range(len(self.num_tasks))
    
    def reset(self, *args, **kwargs):
        self._step = 0
        return  super().reset(*args, **kwargs)

    def reset_task(self, idx):
        self._goal_idx = idx
        self._task = self.tasks[idx]
        self._goal = self._task['goal']
        self.reset()
        
    def get_task(self, ):
        return self._task
    
    def get_idx(self,):
        return self._goal_idx 
    
    def task_modes(self,):
        return {
            'train': np.array(list(range(0,20))),
            'id': np.array(list(range(20, 30))),
            'ood': np.array(list(range(30,40)))
        }
        
    def get_mode(self, ):
        idx = self._goal_idx
        for k,v in self.task_modes().items():
            if idx in v:
                return k
