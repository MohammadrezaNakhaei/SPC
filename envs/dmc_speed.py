from .wrappers import *
from dm_control.suite.wrappers import action_scale
from .contextual_control_suite import suite 


def _make_env(domain, speed, seed=None):
    if domain in ('cheetah', 'walker'):
        task = 'run'
    elif domain in ('finger'):
        task = 'spin'
    else:
        raise NotImplementedError
    
    # Reward parameters
    reward_kwargs = {
        'ALL': {
            'sigmoid': 'linear',
            'margin': speed,
        },
    }

    task_kwargs = {
        'random': seed,
        'reward_kwargs': reward_kwargs,
    }
    env = suite.load(domain,
                        task,
                        task_kwargs=task_kwargs,
                        visualize_reward=False)
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, 2)
    env = action_scale.Wrapper(env, minimum=-1., maximum=1.)
    env = ExtendedTimeStepWrapper(env)
    env = TimeStepToGymWrapper(env, domain, task)
    return env


class SpeedDMC():
    """ Base Class For Reward-Dynamic changing envs"""
    def __init__(self, idx=0, seed=None, **kwargs):
        self.num_tasks = 40
        self.num_train = 20
        self.num_eval_id = 10
        self.num_eval_ood = 10
        self._seed = seed
        self.tasks = self.sample_tasks()
        self.reset_task(idx)

    def sample_tasks(self,):
        # should be defined by descendents
        pass


    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal_idx = idx
        self._env = _make_env(self.env_name, self._task['speed'], self._seed)
        
    def get_all_task_idx(self):
        return range(self.num_tasks)
        
    def get_task(self, ):
        return self._task
    
    def get_idx(self,):
        return self._goal_idx 
    
    def task_modes(self,):
        total = np.arange(0, 30)
        evals = np.arange(1, 31, 3)
        trains = np.setdiff1d(total, evals)
        ood = np.arange(30, 40)
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
            
    def __getattr__(self, name):
        return getattr(self._env, name)
    

class WalkerSpeed(SpeedDMC):
    def __init__(self, idx=0, seed=None, **kwargs):
        self.env_name='walker'
        super().__init__(idx, seed, **kwargs)
        
    def sample_tasks(self,):
        # train and in-distribution tasks
        speeds = np.linspace(-5, -3, 10)
        tasks = [dict(speed=speed) for speed in speeds]
        speeds = np.linspace(-1 ,1, 10)
        tasks.extend(dict(speed=speed) for speed in speeds)
        speeds = np.linspace(3, 5, 10)
        tasks.extend(dict(speed=speed) for speed in speeds)
        # OOD tasks
        speeds = np.linspace(-2.5, -1.5, 5)
        tasks.extend(dict(speed=speed) for speed in speeds)
        speeds = np.linspace(1.5, 2.5, 5)
        tasks.extend(dict(speed=speed) for speed in speeds)
        return tasks
    

class CheetahSpeed(SpeedDMC):
    def __init__(self, idx=0, seed=None, **kwargs):
        self.env_name='cheetah'
        super().__init__(idx, seed, **kwargs)

    def sample_tasks(self,):
        # train and in-distribution tasks
        speeds = np.linspace(-10, -6, 10)
        tasks = [dict(speed=speed) for speed in speeds]
        speeds = np.linspace(-2, 2, 10)
        tasks.extend(dict(speed=speed) for speed in speeds)
        speeds = np.linspace(6, 10, 10)
        tasks.extend(dict(speed=speed) for speed in speeds)
        # OOD tasks
        speeds = np.linspace(-5, -3, 5)
        tasks.extend(dict(speed=speed) for speed in speeds)
        speeds = np.linspace(3, 5, 5)
        tasks.extend(dict(speed=speed) for speed in speeds)
        return tasks
    

class FingerSpeed(SpeedDMC):
    def __init__(self, idx=0, seed=None, **kwargs):
        self.env_name='finger'
        super().__init__(idx, seed, **kwargs)
        
    def sample_tasks(self,):
        # train and in-distribution tasks
        speeds = np.linspace(-15, -9, 10)
        tasks = [dict(speed=speed) for speed in speeds]
        speeds = np.linspace(-3, 3, 10)
        tasks.extend(dict(speed=speed) for speed in speeds)
        speeds = np.linspace(9, 15, 10)
        tasks.extend(dict(speed=speed) for speed in speeds)
        # OOD tasks
        speeds = np.linspace(-8, -4, 5)
        tasks.extend(dict(speed=speed) for speed in speeds)
        speeds = np.linspace(4, 8, 5)
        tasks.extend(dict(speed=speed) for speed in speeds)
        return tasks