#!/bin/bash

nohup python main.py --object_name 'elastic_pendulum' --latent_dyn 'vp' --latent_dim 8 --gpu_index 0 --seed 0 > lep0.file 2>&1 &

