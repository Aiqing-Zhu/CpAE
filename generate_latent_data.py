import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
class NeuralPhysDataset_Determined(Dataset):
    def __init__(self, data_filepath='training_data/', object_name="elastic_pendulum"):
        self.object_name = object_name
        self.data_filepath = data_filepath
        self.all_filelist = self.get_all_filelist()

    def get_all_filelist(self):
        filelist = []
        obj_filepath = os.path.join(self.data_filepath, self.object_name)
        if self.object_name =="elastic_pendulum" or self.object_name =="single_pendulum":
            vid_list = np.arange(0,len(os.listdir(obj_filepath))-1)
        else:
            print('len', len(os.listdir(obj_filepath)))
            vid_list = np.arange(0,len(os.listdir(obj_filepath)))
        for vid_idx in vid_list:
            seq_filepath = os.path.join(obj_filepath, str(vid_idx))
            num_frames = 60#len(os.listdir(seq_filepath))
            
            # if num_frames!=1212:
            #     print(os.listdir(seq_filepath),vid_idx)
            #     print('num_frames', num_frames, seq_filepath)
            suf = os.listdir(seq_filepath)[0].split('.')[-1]
            for p_frame in range(num_frames - 1):
                par_list = []
                for p in range(2):
                    par_list.append(os.path.join(seq_filepath, str(p_frame + p) + '.' + suf))
                filelist.append(par_list)
        return filelist

    def __len__(self):
        return len(self.all_filelist)

    # 0, 1 -> 2, 3
    def __getitem__(self, idx):
        par_list = self.all_filelist[idx]
        # data = self.get_data(par_list[0])
        data = []
        for i in range(2):
            data.append(self.get_data(par_list[i])) # 0, 1
        data = torch.cat(data, 2)
        filepath = '_'.join(par_list[0].split('/')[-2:])
        return data, filepath
    
    def get_data(self, filepath):
        data = Image.open(filepath)
        data = data.resize((128, 128))
        data = np.array(data)
        data = torch.tensor(data / 255.0)
        data = data.permute(2, 0, 1).float()
        return data

from elastic_pendulum_data import mkdir
from tqdm import tqdm


##############

##############

def generate_latent_data(net, data_filepath, object_name="elastic_pendulum", gpu_index=0):
    data = NeuralPhysDataset_Determined(object_name=object_name)
    n=10
    
    if object_name=="swingstick":
        generate_latent_SS_data(net, data_filepath, gpu_index)
    else:
        dataloader = DataLoader(data,
                                batch_size=59*n,
                                shuffle=False,
                                num_workers=0)

        mkdir(data_filepath)
        i=0
        for inputs, filepath in tqdm(dataloader):
            latent=net.encoder(inputs.to('cuda:{}'.format(gpu_index))).squeeze().cpu().detach().numpy()
            for j in range(n):
                np.save(os.path.join(data_filepath, str(i)), latent[j*59:(j+1)*59])
                i=i+1

def generate_latent_SS_data(net, data_filepath, gpu_index=0):
    data = NeuralPhysDataset_Determined(object_name="swingstick")
    dataloader = DataLoader(data,
                            batch_size=173,
                            shuffle=False,
                            num_workers=0)
    mkdir(data_filepath)
    # i=0
    # for inputs, filepath in tqdm(dataloader):
    #     inp = inputs.to('cuda:{}'.format(gpu_index))
    #     latent=net.encoder(inp)
    #     latent_np = latent.squeeze().cpu().detach().numpy()
    #     np.save(os.path.join(data_filepath, str(i)), latent_np)
    #     i=i+1

    latent_np_list=[]
    for inputs, filepath in tqdm(dataloader):
        inp = inputs.to('cuda:{}'.format(gpu_index))
        latent=net.encoder(inp)
        latent_np = latent.squeeze().cpu().detach().numpy()
        latent_np_list.append(latent_np)
        
        
    for i in range(85):
        latent_np = np.vstack(latent_np_list[i*7:i*7+7])
        np.save(os.path.join(data_filepath, str(i)), latent_np)
