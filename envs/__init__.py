from argparse import Action
from .ant_dir import AntDirEnvCustom 
from .ant_goal import AntGoalEnvCustom 
from .cheetah_vel import HalfCheetahVelEnvCustom
from .hopper_params import HopperParamsMass, HopperParamsFriction
from .walker_params import WalkerParamsMass, WalkerParamsFriction
from .wrappers import NormalizedBoxEnv
from .point_robot import PointRobot
from .dmc_reward_dynamics import WalkerLS, CheetahLS, FingerLS
from .dmc_speed import CheetahSpeed, FingerSpeed, WalkerSpeed
from .metaworld import MetaWorldEnv

from .metaworld import MetaWorldEnv


ENVS = {
    'ant-dir': AntDirEnvCustom,
    'ant-goal': AntGoalEnvCustom,
    'cheetah-vel': HalfCheetahVelEnvCustom,
    'hopper-mass': HopperParamsMass,
    'hopper-friction': HopperParamsFriction,
    'walker-mass': WalkerParamsMass,
    'walker-friction': WalkerParamsFriction,
    'point-robot': PointRobot,
    'walker-speed': WalkerSpeed,
    'cheetah-speed': CheetahSpeed,
    'finger-speed': FingerSpeed,
}

DMENVS = {
    'walker-ls': WalkerLS,
    'cheetah-ls': CheetahLS,
    'finger-ls': FingerLS,
}



def make_meta_env(env_id, idx=0, height=256, width=256, seed=None):
    if env_id in ENVS:
        return NormalizedBoxEnv(ENVS[env_id](render_mode='rgb_array', idx=idx,height=height, width=width))
    elif env_id in DMENVS:
        return DMENVS[env_id](idx, seed=seed)
    elif env_id.startswith('mt-'):
        return MetaWorldEnv(env_id[3:], idx=idx, seed=seed)
    else:
        raise NotImplementedError
    