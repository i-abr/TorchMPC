#!/bin/sh

for seed in 0 1 42 666 1234
do
    python3 main.py \
                --env "Walker2DEnv" \
                --max_steps 2000 \
                --max_frames 80000 \
		        --method 'mppi' \
                --horizon 10 \
                --frame_skip 4 \
                --seed $seed \
		        --log \
                --no_render
    echo "trial $seed"
done
