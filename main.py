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

parser.add_argument('--seed', type=int, default=666)

parser.add_argument('--horizon', type=int, default=5)
parser.add_argument('--model_iter', type=int, default=2)

parser.add_argument('--method', type=str, default='mppi')

parser.add_argument('--done_util', dest='done_util', action='store_true')
parser.add_argument('--no_done_util', dest='done_util', action='store_false')
parser.set_defaults(done_util=True)

parser.add_argument('--render', dest='render', action='store_true')
parser.add_argument('--no_render', dest='render', action='store_false')
parser.set_defaults(render=False)

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


    if args.log:
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d_%H-%M-%S/")
        dir_name = 'seed_{}/'.format(args.seed)
        path = './data/' + args.method +  '/' + args.env + '/' + dir_name
        if os.path.exists(path) is False:
            os.makedirs(path)

    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]

    device ='cpu'
    print(torch.cuda.is_available())
    input()
    if torch.cuda.is_available():
        device  = 'cuda:0'
        print('Using GPU Accel')

    model       = Model(state_dim, action_dim, def_layers=[200]).to(device)
    # model = MDNModel(state_dim, action_dim, def_layers=[200, 200])

    replay_buffer_size = 100000
    model_replay_buffer = SARSAReplayBuffer(replay_buffer_size)
    model_optim = ModelOptimizer(model, model_replay_buffer, lr=args.model_lr)
    # model_optim = MDNModelOptimizer(model, replay_buffer, lr=args.model_lr)

    methods = {'ilqr' : iLQR, 'shooting': ShootingMethod, 'mppi' : MPPI}

    # mpc_planner = iLQR(model, T=args.horizon)
    # mpc_planner = ShootingMethod(model, T=args.horizon)
    # mpc_planner = MPPI(model, T=args.horizon)
    mpc_planner = methods[args.method](model, T=args.horizon)

    max_frames  = args.max_frames
    max_steps   = args.max_steps
    frame_skip = args.frame_skip

    frame_idx   = 0
    rewards     = []
    batch_size  = 256

    ep_num = 0
    while frame_idx < max_frames:
        state = env.reset()
        mpc_planner.reset()

        action = mpc_planner.update(state)

        episode_reward = 0
        done = False
        for step in range(max_steps):
            for _ in range(frame_skip):
                next_state, reward, done, _ = env.step(action.copy())

            next_action = mpc_planner.update(next_state)
            # next_action = policy_net.get_action(next_state)
            if args.method == 'ilqr' or args.method == 'shooting':
                eps = 1.0 * (0.995**frame_idx)
                next_action = next_action + np.random.normal(0., eps, size=(action_dim,))

            model_replay_buffer.push(state, action, reward, next_state, next_action, done)

            if len(model_replay_buffer) > batch_size:
                model_optim.update_model(batch_size, mini_iter=args.model_iter)

            state = next_state
            action = next_action
            episode_reward += reward
            frame_idx += 1

            if args.render:
                env.render("human")


            if frame_idx % (max_frames//10) == 0:
                last_reward = rewards[-1][1] if len(rewards)>0 else 0
                print(
                    'frame : {}/{}, \t last rew: {}'.format(
                        frame_idx, max_frames, last_reward
                    )
                )

                if args.log:
                    print('saving model and reward')
                    pickle.dump(rewards, open(path + 'reward_data' + '.pkl', 'wb'))
                    torch.save(model.state_dict(), path + 'model_' + str(frame_idx) + '.pt')

            if args.done_util:
                if done:
                    break

        print('ep rew', ep_num, episode_reward)
        rewards.append([frame_idx, episode_reward])
        ep_num += 1

    if args.log:
        print('saving final data set')
        pickle.dump(rewards, open(path + 'reward_data' + '.pkl', 'wb'))
        torch.save(model.state_dict(), path + 'model_' + 'final' + '.pt')
    # pickle.dump(rewards, open(path + 'reward_data'+ '.pkl', 'wb'))
    # torch.save(policy_net.state_dict(), path + 'policy_' + 'final' + '.pt')
    # torch.save(model.state_dict(), path + 'model_' + 'final' + '.pt')
