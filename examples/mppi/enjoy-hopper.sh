#!/bin/sh

for seed in 0 1 42 666 1234
do
    python3 enjoy.py \
                --env "HopperBulletEnv" \
                --max_steps 2000 \
                --max_frames 80000 \
		        --method 'mppi' \
                --horizon 10 \
                --frame_skip 4 \
                --seed $seed \
                --render \
                --no-record \
                --file_path "./data/mppi/HopperBulletEnv/seed_$seed/model_final.pt"
    echo "trial $seed"
done
