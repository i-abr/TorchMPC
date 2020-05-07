#!/bin/zsh

for i in {1..2}
do
    python3 h_ddpg.py \
                --env "HopperBulletEnv" \
                --max_steps 400 \
                --max_frames 10000 \
                --horizon 5 \
                --frame_skip 4 \
                --no_render \
                --jacobi_weight 1e-1 \
                --ctrl_weight 10 \
                --model_iter 2 \
                --model_lr 3e-3 \
                --value_lr 3e-4 \
                --policy_lr 3e-4
    echo "trial $i out of 2"
done
