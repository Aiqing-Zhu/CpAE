#!/bin/bash

nohup python main.py --object_name 'single_pendulum' --latent_dyn 'vp' --latent_dim 2 --gpu_index 0 --seed 0 > lsp0.file 2>&1 &


