#!/bin/zsh

for seed in 0 1 42 666 1234
do
    python3 hlt_stoch.py \
                --env 'InvertedPendulumSwingupBulletEnv' \
                --max_frames 10000 \
                --frame_skip 4 \
                --max_steps 1000 \
                --lam 0.1 \
                --model_iter 5 \
                --seed $seed \
                --render
    echo "trial $i out of $trials"
done
