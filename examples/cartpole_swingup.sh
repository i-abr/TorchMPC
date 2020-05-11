#!/bin/zsh

for seed in 0 1 42 666 1234
do
    python3 main.py \
                --env 'InvertedPendulumSwingupBulletEnv' \
                --method 'mppi' \
                --max_frames 10000 \
                --frame_skip 4 \
                --horizon 5 \
                --max_steps 1000 \
                --model_iter 5 \
                --seed $seed \
                --log \
                --render
    echo "trial $i out of $trials"
done
