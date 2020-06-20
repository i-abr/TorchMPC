#!/bin/sh

for seed in 0 1 42 666 1234
do
    python3 enjoy.py \
                --env "AntBulletEnv" \
                --max_steps 2000 \
                --max_frames 100000 \
		        --method 'mppi' \
                --horizon 10 \
                --frame_skip 5 \
                --seed $seed \
                --render \
                --no-record \
                --file_path "./data/mppi/AntBulletEnv/seed_$seed/model_final.pt"
    echo "trial $seed"
done
