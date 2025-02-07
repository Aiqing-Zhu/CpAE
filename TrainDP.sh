#!/bin/bash

nohup python main.py --object_name 'double_pendulum' --latent_dyn 'vp' --latent_dim 8 --gpu_index 2 --seed 0 > ldp0.file 2>&1 &


