#!/bin/sh

for seed in 0 1 42 666 1234
do
    python3 main.py \
                --env "HalfCheetahBulletEnv" \
                --max_steps 2000 \
                --max_frames 40000 \
		--method 'shooting' \
                --horizon 10 \
                --frame_skip 5 \
                --seed $seed \
		--log \
                --no_render
    echo "trial $seed"
done
