#!/bin/sh

for seed in 666
do
    python3 enjoy.py \
                --env 'InvertedPendulumSwingupBulletEnv' \
                --method 'shooting' \
                --max_frames 2000 \
                --frame_skip 4 \
                --horizon 5 \
                --max_steps 1000 \
                --model_iter 5 \
                --seed $seed \
                --render \
                --file_path ./data/shooting/InvertedPendulumSwingupBulletEnv/2020-05-12_21-51-32/model_final.pt
    echo "trial $seed"
done
