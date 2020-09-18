from pybullet_envs import gym_pendulum_envs, gym_manipulator_envs, gym_locomotion_envs
from gym.envs import classic_control, box2d, mujoco
from pybullet_envs.bullet.racecarGymEnv import RacecarGymEnv
from pybullet_envs.bullet.minitaur_gym_env import MinitaurBulletEnv

# from rex_gym.envs.gym.galloping_env import RexReactiveEnv

env_list = {
    'InvertedPendulumBulletEnv' : gym_pendulum_envs.InvertedPendulumSwingupBulletEnv,
    'HalfCheetah' : mujoco.HalfCheetahEnv,
    #'HalfCheetahBulletEnv' : gym_locomotion_envs.HalfCheetahBulletEnv,
    'HopperBulletEnv' : gym_locomotion_envs.HopperBulletEnv,
    'AntBulletEnv' : gym_locomotion_envs.AntBulletEnv,
    'ReacherBulletEnv' : gym_manipulator_envs.ReacherBulletEnv,
    'PendulumEnv' : classic_control.PendulumEnv,
    'Walker2DEnv' : gym_locomotion_envs.Walker2DBulletEnv,
}

def getlist():
    out_str = ''
    for env_name in env_list.keys():
        out_str += env_name + '\n'
    return out_str
