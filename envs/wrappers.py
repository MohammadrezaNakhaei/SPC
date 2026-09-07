import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box

from collections import deque, defaultdict
from typing import Any, NamedTuple
import dm_env
from dm_control import suite

suite.ALL_TASKS = suite.ALL_TASKS + suite._get_tasks('custom')
suite.TASKS_BY_DOMAIN = suite._get_tasks_by_domain(suite.ALL_TASKS)
from dm_env import StepType, specs
import gymnasium as gym


class NormalizedBoxEnv(Env):
    """
    Normalize action to in [-1, 1].

    """
    def __init__(
            self,
            env,
            reward_scale=1.0,
    ):
        self._wrapped_env = env
        self._reward_scale = reward_scale
        ub = np.ones(self._wrapped_env.action_space.shape)
        self.action_space = Box(-1 * ub, ub)

    def step(self, action):
        lb = self._wrapped_env.action_space.low
        ub = self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)

        wrapped_step = self._wrapped_env.step(scaled_action)
        next_obs, reward, terminated, truncated, info = wrapped_step
        return next_obs, reward*self._reward_scale, terminated, truncated, info

    def __str__(self):
        return "Normalized: %s" % self._wrapped_env

    def __getattr__(self, attrname):
        return getattr(self._wrapped_env, attrname)

    def reset(self, *args, **kwargs):
        return self._wrapped_env.reset(*args, **kwargs)
    
    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)


class D4RlGymnasium(Env):
    """
    Normalize action to in [-1, 1].

    Optionally normalize observations and scale reward.
    """
    def __init__(
            self,
            env,
            normalize_reward=True,
            height=256,
            width=256,
    ):
        self._wrapped_env = env
        self._nomralize_reward = normalize_reward
        self.height = height
        self.width = width
        self.observation_space = gym.spaces.Box(
            dtype=np.float32, 
            shape=self.observation_space.shape, 
            low=self.observation_space.low, 
            high=self.observation_space.high)
        self.action_space = gym.spaces.Box(low=-1, high=+1, shape=self.action_space.shape, dtype=np.float32)

    def step(self, action):
        wrapped_step = self._wrapped_env.step(action)
        next_obs, reward, done, info = wrapped_step
        if self._nomralize_reward:
            reward = self.get_normalized_score(reward)
        self._step += 1
        truncated = self._step>=self._max_episode_steps
        return next_obs, reward, done, truncated, info

    def __str__(self):
        return "D4RLToGymnasium: %s" % self._wrapped_env

    def __getattr__(self, attrname):
        return getattr(self._wrapped_env, attrname)
    
    def reset(self, *args, **kwargs):
        obs = self._wrapped_env.reset()
        self._step = 0
        return obs, {}
    
    def render(self, ):
        return self._wrapped_env.render(mode='rgb_array', height=self.height, width=self.width)
	

class ExtendedTimeStep(NamedTuple):
	step_type: Any
	reward: Any
	discount: Any
	observation: Any
	action: Any

	def first(self):
		return self.step_type == StepType.FIRST

	def mid(self):
		return self.step_type == StepType.MID

	def last(self):
		return self.step_type == StepType.LAST


class ActionRepeatWrapper(dm_env.Environment):
	def __init__(self, env, num_repeats):
		self._env = env
		self._num_repeats = num_repeats

	def step(self, action):
		reward = 0.0
		discount = 1.0
		for i in range(self._num_repeats):
			time_step = self._env.step(action)
			reward += (time_step.reward or 0.0) * discount
			discount *= time_step.discount
			if time_step.last():
				break

		return time_step._replace(reward=reward, discount=discount)

	def observation_spec(self):
		return self._env.observation_spec()

	def action_spec(self):
		return self._env.action_spec()

	def reset(self):
		return self._env.reset()

	def __getattr__(self, name):
		return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
	def __init__(self, env, dtype):
		self._env = env
		wrapped_action_spec = env.action_spec()
		self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
											   dtype,
											   wrapped_action_spec.minimum,
											   wrapped_action_spec.maximum,
											   'action')

	def step(self, action):
		action = action.astype(self._env.action_spec().dtype)
		return self._env.step(action)

	def observation_spec(self):
		return self._env.observation_spec()

	def action_spec(self):
		return self._action_spec

	def reset(self):
		return self._env.reset()

	def __getattr__(self, name):
		return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
	def __init__(self, env):
		self._env = env

	def reset(self):
		time_step = self._env.reset()
		return self._augment_time_step(time_step)

	def step(self, action):
		time_step = self._env.step(action)
		return self._augment_time_step(time_step, action)

	def _augment_time_step(self, time_step, action=None):
		if action is None:
			action_spec = self.action_spec()
			action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
		return ExtendedTimeStep(observation=time_step.observation,
								step_type=time_step.step_type,
								action=action,
								reward=time_step.reward or 0.0,
								discount=time_step.discount or 1.0)

	def observation_spec(self):
		return self._env.observation_spec()

	def action_spec(self):
		return self._env.action_spec()

	def __getattr__(self, name):
		return getattr(self._env, name)


class TimeStepToGymWrapper:
	def __init__(self, env, domain, task, width=256, height=256, camera_id=0):
		obs_shp = []
		for v in env.observation_spec().values():
			try:
				shp = np.prod(v.shape)
			except:
				shp = 1
			obs_shp.append(shp)
		obs_shp = (int(np.sum(obs_shp)),)
		act_shp = env.action_spec().shape
		self.observation_space = gym.spaces.Box(
			low=np.full(
				obs_shp,
				-np.inf,
				dtype=np.float32),
			high=np.full(
				obs_shp,
				np.inf,
				dtype=np.float32),
			dtype=np.float32,
		)
		self.action_space = gym.spaces.Box(
			low=np.full(act_shp, env.action_spec().minimum),
			high=np.full(act_shp, env.action_spec().maximum),
			dtype=env.action_spec().dtype)
		self.env = env
		self.domain = domain
		self.task = task
		self.max_episode_steps = 500
		self.t = 0
		self.width = width 
		self.height = height 
		camera_id = dict(quadruped=2).get(self.domain, camera_id)
		self.camera_id = camera_id
		self.render_mode = 'rgb_array'
	
	@property
	def unwrapped(self):
		return self.env

	@property
	def reward_range(self):
		return None

	@property
	def metadata(self):
		return dict(render_modes=['rgb_array'], autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)
	
	def _obs_to_array(self, obs):
		return np.concatenate([v.flatten() for v in obs.values()])

	def reset(self, *args, **kwargs):
		self.t = 0
		return self._obs_to_array(self.env.reset().observation), dict()
	
	def step(self, action):
		self.t += 1
		time_step = self.env.step(action)
		return self._obs_to_array(time_step.observation), time_step.reward, False, time_step.last() or self.t == self.max_episode_steps, defaultdict(float)

	def render(self):
		return self.env.physics.render(self.height, self.width, self.camera_id)
	
	def close(self,):
		pass

class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)





