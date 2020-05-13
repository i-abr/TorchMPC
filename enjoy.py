import numpy as np
import random
import pickle
from datetime import datetime

import sys
import os

# local imports
import envs

import torch
from mpc_lib import iLQR
from mpc_lib import ShootingMethod
from mpc_lib import MPPI

from model import ModelOptimizer, Model, SARSAReplayBuffer
from normalized_actions import NormalizedActions
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env',        type=str,   help=envs.getlist())
parser.add_argument('--max_steps',  type=int,   default=200)
parser.add_argument('--max_frames', type=int,   default=10000)
parser.add_argument('--frame_skip', type=int,   default=2)
parser.add_argument('--model_lr',   type=float, default=3e-4)
parser.add_argument('--policy_lr',  type=float, default=3e-4)
parser.add_argument('--file_path', type=str, default='none')


parser.add_argument('--seed', type=int, default=666)

parser.add_argument('--horizon', type=int, default=5)
parser.add_argument('--model_iter', type=int, default=2)

parser.add_argument('--method', type=str, default='shooting')

parser.add_argument('--done_util', dest='done_util', action='store_true')
parser.add_argument('--no_done_util', dest='done_util', action='store_false')
parser.set_defaults(done_util=True)

parser.add_argument('--render', dest='render', action='store_true')
parser.add_argument('--no_render', dest='render', action='store_false')
parser.set_defaults(render=True)

parser.add_argument('--log', dest='log', action='store_true')
parser.add_argument('--no-log', dest='log', action='store_false')
parser.set_defaults(log=False)
args = parser.parse_args()


if __name__ == '__main__':


    env_name = args.env
    try:
        env = NormalizedActions(envs.env_list[env_name](render=args.render))
    except TypeError as err:
        print('no argument render,  assumping env.render will just work')
        env = NormalizedActions(envs.env_list[env_name]())
    assert np.any(np.abs(env.action_space.low) <= 1.) and  np.any(np.abs(env.action_space.high) <= 1.), 'Action space not normalizd'
    env.reset()

    env.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]

    device ='cpu'
    if torch.cuda.is_available():
        device  = 'cuda:0'
        print('Using GPU Accel')

    model = Model(state_dim, action_dim, def_layers=[200]).to(device)
    # state_dict_dir = './data/'+args.method+'/seed{}/'.format(args.seed) + args.file_name
    state_dict_path = args.file_path
    model.load_state_dict(torch.load(state_dict_path, map_location=device))


    methods = {'ilqr' : iLQR, 'shooting': ShootingMethod, 'mppi' : MPPI}
    mpc_planner = methods[args.method](model, T=args.horizon)

    max_frames  = args.max_frames
    max_steps   = args.max_steps
    frame_skip = args.frame_skip

    frame_idx   = 0
    rewards     = []

    ep_num = 0
    while frame_idx < max_frames:
        state = env.reset()
        mpc_planner.reset()

        episode_reward = 0
        done = False
        for step in range(max_steps):

            action = mpc_planner.update(state)
            for _ in range(frame_skip):
                state, reward, done, _ = env.step(action.copy())
            episode_reward += reward
            frame_idx += 1

            if args.render:
                env.render("human")

            if args.done_util:
                if done:
                    break

        print('ep rew', ep_num, episode_reward)
        rewards.append([frame_idx, episode_reward])
        ep_num += 1
