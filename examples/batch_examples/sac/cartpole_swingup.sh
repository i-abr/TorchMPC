#!/bin/zsh

for seed in 0 1 42 666 1234
do
    python3 sac.py \
                --env 'InvertedPendulumSwingupBulletEnv' \
                --max_frames 20000 \
                --frame_skip 4 \
                --max_steps 1000 \
                --seed $seed \
                --no_render
    echo "trial $i out of $trials"
done
