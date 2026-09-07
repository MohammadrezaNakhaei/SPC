import numpy as np 
from gymnasium.envs.mujoco.hopper_v5 import HopperEnv as HopperEnv_


class HopperENV(HopperEnv_):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._max_episode_steps = 200
        self._step = 0
                
    def step(self, a):
        posbefore = self.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        terminated = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        # done = False
        self._step += 1
        truncated = True if self._step >= self._max_episode_steps else False
        ob = self._get_obs()
        return ob, reward, terminated, truncated, {}

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat[1:],
            np.clip(self.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()
    
    def reset(self, *args, **kwargs):
        self._step = 0
        return super().reset(*args, **kwargs)
    

class HopperParamsMass(HopperENV):
    def __init__(self,idx=0, **kwargs):
        super().__init__(**kwargs)
        self.original_mass = np.copy(self.model.body_mass)
        self.original_inertia = np.copy(self.model.body_inertia)
        self.num_tasks = 40
        self.num_train = 20
        self.num_eval_id = 10
        self.num_eval_ood = 10
        self.tasks = self.sample_tasks()
        self._max_episode_steps = 200
        self._step = 0
        self.reset_task(idx)
        
    def sample_tasks(self, ):
        logscales = np.linspace(-2, 2.0, num=self.num_tasks)
        tasks = [{'scale': 1.5**logscale} for logscale in logscales]
        return tasks

    def get_all_task_idx(self):
        return range(self.num_tasks)

    def reset_task(self, idx):
        self._goal_idx = idx
        self._task = self.tasks[idx]
        self._goal = self._task['scale']
        self.model.body_mass[:] = self.original_mass*self._goal
        self.model.body_inertia[:] = self.original_inertia*self._goal
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


class HopperParamsFriction(HopperENV):
    def __init__(self,idx=0, **kwargs):
        super().__init__(**kwargs)
        self.original_friction = np.copy(self.model.geom_friction)
        self.num_tasks = 40
        self.num_train = 20
        self.num_eval_id = 10
        self.num_eval_ood = 10
        self.tasks = self.sample_tasks()
        self._max_episode_steps = 200
        self._step = 0
        self.reset_task(idx)

    def sample_tasks(self, ):
        logscales = np.linspace(-2, 2.0, num=self.num_tasks)
        tasks = [{'scale': 1.5**logscale} for logscale in logscales]
        return tasks

    def get_all_task_idx(self):
        return range(self.num_tasks)

    def reset_task(self, idx):
        self._goal_idx = idx
        self._task = self.tasks[idx]
        self._goal = self._task['scale']
        self.model.geom_friction[:] = self.original_friction*self._goal
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
