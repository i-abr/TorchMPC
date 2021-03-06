#!/bin/zsh

for i in {1..1}
do
    python3 hd_sac.py \
                --env "HopperBulletEnv" \
                --max_steps 1000 \
                --max_frames 40000 \
                --horizon 5 \
                --frame_skip 4 \
                --no_render
    echo "trial $i out of 2"
done
