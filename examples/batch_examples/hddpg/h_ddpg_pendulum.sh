#!/bin/zsh
trials=2
for i in {1..$trials}
do
    python3 h_ddpg.py --env 'PendulumEnv' --max_frames 6000 --ctrl_weight 1e-2 --frame_skip 4 --render
    echo "trial $i out of $trials"
done
