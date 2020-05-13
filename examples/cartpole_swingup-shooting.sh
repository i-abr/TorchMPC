#!/bin/sh

for seed in 0 1 42 666 1234
do
    python3 main.py \
                --env 'InvertedPendulumSwingupBulletEnv' \
                --method 'shooting' \
                --max_frames 20000 \
                --frame_skip 4 \
                --horizon 5 \
                --max_steps 1000 \
                --model_iter 5 \
                --seed $seed \
                --log \
                --no_render
    echo "trial $seed"
done
