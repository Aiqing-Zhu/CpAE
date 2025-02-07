# PyTorch Implementation of: Continuity-Preserving Convolutional Autoencoders for Learning Continuous Latent Dynamical Models from Images

This repository contains code to reproduce the results as presented in the paper: Continuity-Preserving Convolutional Autoencoders for Learning Continuous Latent Dynamical Models from Images.

## Requirements 
* `environment.yml`

## Reproducing the Results of the Paper
All necessary parameters are specified in the paper. The code for our experiments is implemented using the [PyTorch framework](https://openreview.net/pdf?id=BJJsrmfCZ) and executed on RTX 3090 GPUs.

### Running Experiments for Circular Motion
```
python Test.py
```

### Running Experiments for Single Pendulum and Elastic Double Pendulum
To generate data, run:
```
make_data.ipynb
```
After generating the data, run:
```
sh TrainSP.sh
sh TrainEP.sh
```

### Running Experiments for Double Pendulum and Swing Stick
#### Data Preparation
The real-world data are provided by:

[Boyuan Chen, Kuang Huang, Sunand Raghupathi, Ishaan Chandratreya, Qiang Du, and Hod Lipson. Automated discovery of fundamental variables hidden in experimental data. Nature Computational Science, 2(7):433–442, 2022.](https://www.cs.columbia.edu/~bchen/neural-state-variables/)

The datasets can be downloaded via the following links:

- [Double Pendulum](https://drive.google.com/file/d/1QEtk4JjnRysIEjtkZIKBjACu_IiTAdX6/view?usp=sharing) (rigid double pendulum system)
- [Swing Stick (Non-Magnetic)](https://drive.google.com/file/d/1BfeGW4XTFyGdyBO0G_YnSnRyJGu2WRnc/view?usp=sharing) (swing stick system)

Save the downloaded datasets under the directory: `training_data/{dataset_name}`.

#### Training
To start training, run:
```
sh TrainDP.sh
sh TrainSS.sh
```

## Reference

Aiqing Zhu, Yuting Pan, Qianxiao Li. Continuity-Preserving Convolutional Autoencoders for Learning Continuous Latent Dynamical Models from Images. The Thirteenth International Conference on Learning Representations (ICLR 2025).
 
