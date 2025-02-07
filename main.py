import numpy as np
import torch

import dynlearner as ln
from AE_model import CAE_hyp
from dataset import CudaPhysDataset, PhysDataset, NeuralPhysDataset


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_index',type=int, default=0)
parser.add_argument('--seed',type=int, default=0)
parser.add_argument('--object_name',type=str, default='elastic_pendulum')
parser.add_argument('--latent_dyn',type=str, default='vp')
parser.add_argument('--latent_dim',type=int, default=8)

args = parser.parse_args()

def main(object_name='elastic_pendulum', latent_dyn='vp', latent_dim=6, gpu_index=2, seed=2):
    device = 'gpu' # 'cpu' or 'gpu'
    torch.cuda.set_device(gpu_index)

    train_dataset = NeuralPhysDataset(flag='train', train=True, object_name=object_name)
    test_dataset = NeuralPhysDataset(flag='val', train=True, object_name=object_name)#

    filename = object_name + '_caehyp_' + latent_dyn+'{}'.format(seed)
    net = CAE_hyp(latent_dim=latent_dim, latent_dyn=latent_dyn, lam=1) 

    print(filename, 'CAE_hyp with' + latent_dyn)
    args = {
        'filename': filename,
        'dataset': [train_dataset, test_dataset],
        'shuffle':True,
        'num_workers':0,
        'net': net,
        'criterion': None,
        'optimizer': 'adam',
        'lr': 0.001,
        'iterations': 3,
        'batch_size': 512,
        'print_every': 1,
        'save': True,
        'callback': None,
        'dtype': 'float',
        'device': device
    }

    ln.ImageBrain.Init(**args)
    ln.ImageBrain.Run()
    ln.ImageBrain.Restore()
    ln.ImageBrain.Output()
    

if __name__ == '__main__': 
    main(object_name=args.object_name, latent_dyn=args.latent_dyn, latent_dim=args.latent_dim, gpu_index=args.gpu_index, seed=args.seed)
    from generate_latent_data import generate_latent_data
    filename=args.object_name + '_caehyp_' + args.latent_dyn+'{}'.format(args.seed)
    path='outputs/'+filename+'/model_best.pkl'
    net = torch.load(path, map_location='cuda:{}'.format(args.gpu_index)).eval()
    
    data_filepath = 'training_data/latent_' + filename
    generate_latent_data(net=net, data_filepath=data_filepath, object_name=args.object_name, gpu_index=args.gpu_index)
    
    from learn_latent_dynamics import train_latent_dynamics
    train_latent_dynamics(dim=args.latent_dim, object_name=args.object_name, latent_dyn=args.latent_dyn, gpu_index=args.gpu_index, seed=args.seed)
    
    