#!/bin/zsh
trials=2
for i in {1..$trials}
do
    python3 h_ddpg.py --env 'LunarLanderContinuousEnv' --max_steps 200 --policy_lr 3e-4 --max_frames 20000 --frame_skip 1 --no_render --horizon 20 --ctrl_weight 1e-3
    echo "trial $i out of $trials"
done
