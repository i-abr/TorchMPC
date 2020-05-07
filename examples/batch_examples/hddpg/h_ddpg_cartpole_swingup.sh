#!/bin/zsh

trials=2
for i in {1..$trials}
do
    python3 h_ddpg.py \
                --env 'InvertedPendulumSwingupBulletEnv' \
                --max_frames 10000 \
                --frame_skip 4 \
                --horizon 10 \
                --jacobi_weight 1e-3 \
                --policy_lr 3e-3 \
                --ctrl_weight 1e-2 \
                --render
    echo "trial $i out of $trials"
done
