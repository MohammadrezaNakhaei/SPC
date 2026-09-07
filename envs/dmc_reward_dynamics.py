from .wrappers import *
from dm_control.suite.wrappers import action_scale
from .contextual_control_suite import suite 


def _make_env(domain, speed, length, seed=None):
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

    # Dynamics parameters
    dynamics_kwargs = {
        'length': length,
    }

    task_kwargs = {
        'random': seed,
        'reward_kwargs': reward_kwargs,
        'dynamics_kwargs': dynamics_kwargs
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


class DMC():
    """ Base Class For Reward-Dynamic changing envs"""
    def __init__(self, idx=0, seed=None, **kwargs):
        self.num_tasks = 38
        self.num_train = 20
        self.num_eval_id = 10
        self.num_eval_ood = 8
        self._seed = seed
        self.tasks = self.sample_tasks()
        self.reset_task(idx)

    def sample_tasks(self,):
        # should be defined by descendents
        pass

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal_idx = idx
        self._env = _make_env(self.env_name, self._task['speed'], self._task['length'], self._seed)
        
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
        ood = np.arange(30, 38)
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
    

class WalkerLS(DMC):
    def __init__(self, idx=0, seed=None, **kwargs):
        self.env_name='walker'
        super().__init__(idx, seed, **kwargs)
        
    def sample_tasks(self,):
        # train and in-distribution tasks
        speeds = np.linspace(2, 4.5, 6)
        lengths = np.linspace(0.2, 0.4, 5)
        tasks = [dict(speed=speed, length=length) for speed in speeds for length in lengths]
        speeds = (1, 1.5)
        lengths = (0.1, 0.15)
        tasks.extend(dict(speed=speed, length=length) for speed in speeds for length in lengths)
        speeds = (5, 5.5)
        lengths = (0.45, 0.5)
        tasks.extend(dict(speed=speed, length=length) for speed in speeds for length in lengths)
        return tasks
    

class CheetahLS(DMC):
    def __init__(self, idx=0, seed=None, **kwargs):
        self.env_name='cheetah'
        super().__init__(idx, seed, **kwargs)

    def sample_tasks(self,):
        # train and in-distribution tasks
        speeds = np.linspace(3, 8.0, 6)
        lengths = np.linspace(0.4, 0.6, 5)
        tasks = [dict(speed=speed, length=length) for speed in speeds for length in lengths]
        speeds = (1, 2)
        lengths = (0.3, 0.35)
        tasks.extend(dict(speed=speed, length=length) for speed in speeds for length in lengths)
        speeds = (9, 10)
        lengths = (0.65, 0.7)
        tasks.extend(dict(speed=speed, length=length) for speed in speeds for length in lengths)
        return tasks
    

class FingerLS(DMC):
    def __init__(self, idx=0, seed=None, **kwargs):
        self.env_name='finger'
        super().__init__(idx, seed, **kwargs)
        
    def sample_tasks(self,):
        # train and in-distribution tasks
        speeds = np.linspace(5, 10.0, 6)
        lengths = np.linspace(0.15, 0.25, 5)
        tasks = [dict(speed=speed, length=length) for speed in speeds for length in lengths]
        speeds = (3, 4)
        lengths = (0.1, 0.125)
        tasks.extend(dict(speed=speed, length=length) for speed in speeds for length in lengths)
        speeds = (11, 12)
        lengths = (0.275, 0.3)
        tasks.extend(dict(speed=speed, length=length) for speed in speeds for length in lengths)
        return tasks