import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader 
import os
from PIL import Image 
import json
from tqdm import tqdm


class NeuralPhysDataset(Dataset):
    def __init__(self, data_filepath='training_data/', flag='train',step_number=3, seed=1, object_name="elastic_pendulum", train=False):
        self.seed = seed
        self.flag = flag
        self.scale=1 if train else 255
        self.step_number = step_number
        self.object_name = object_name
        self.data_filepath = data_filepath
        self.all_filelist = self.get_all_filelist()

    def get_all_filelist(self):
        filelist = []
        obj_filepath = os.path.join(self.data_filepath, self.object_name)
        # get the video ids based on training or testing data
        with open(os.path.join('./datainfo', self.object_name, f'data_split_dict_{self.seed}.json'), 'r') as file:
            seq_dict = json.load(file)

        vid_list = seq_dict[self.flag] 
        # go through all the selected videos and get the triplets: input(t, t+1), output(t+2)
        
        length = len(os.listdir(os.path.join(obj_filepath, str(0))))
        for vid_idx in vid_list:
            seq_filepath = os.path.join(obj_filepath, str(vid_idx))
            num_frames = len(os.listdir(seq_filepath))
            if num_frames!=length:
                print(num_frames,length)
                print(os.listdir( os.path.join(obj_filepath, str(0))),0)
                print(os.listdir(seq_filepath),vid_idx)
                raise ValueError("wrong data")
            suf = os.listdir(seq_filepath)[0].split('.')[-1]
            for p_frame in range(num_frames - self.step_number):
                par_list = []
                for p in range(self.step_number+1):
                    par_list.append(os.path.join(seq_filepath, str(p_frame + p) + '.' + suf))
                filelist.append(par_list)
        return filelist

    def __len__(self):
        return len(self.all_filelist)

    # 0, 1 -> 2, 3 
    def __getitem__(self, idx):
        par_list = self.all_filelist[idx]
        data = []
        for i in range(2):
            data.append(self.get_data(par_list[i])) # 0, 1
        data = torch.cat(data, 2)
        target = []
        target.append(self.get_data(par_list[-2])) # 2
        target.append(self.get_data(par_list[-1])) # 3
        target = torch.cat(target, 2)
        filepath = '_'.join(par_list[0].split('/')[-2:])
        return data, target#, filepath, par_list       

    def get_data(self, filepath):
        data = Image.open(filepath)
        data = data.resize((128, 128))
        data = np.array(data)
        data = torch.tensor(data)/self.scale
        data = data.permute(2, 0, 1).float()
        return data


class CudaPhysDataset(Dataset):
    def __init__(self, dataset, Device='cuda'):
        self.dataset=dataset
        self.Device = Device
        self.all_filelist = dataset.all_filelist
        self.X, self.y=self.get_all_data()
        

    def get_all_data(self):
        dataloader = DataLoader(self.dataset,
                                batch_size=512,
                                shuffle=False,
                                num_workers=0)
        X_list=[]
        y_list=[]
        
        print('get all data')
        for inputs, targets in tqdm(dataloader):
            X_list.append(inputs.to(self.Device))
            y_list.append(targets.to(self.Device))

        X = torch.concatenate(X_list, axis=0)
        y = torch.concatenate(y_list, axis=0)
        return X, y
    
    def __len__(self):
        return len(self.X)

    # 0, 1 -> 2, 3 
    def __getitem__(self, idx): 
        return self.X[idx], self.y[idx]

        
class PhysDataset(Dataset):
    def __init__(self, data_filepath='training_data/', flag='train',step_number=3, seed=1, object_name="elastic_pendulum"):
        self.seed = seed
        self.flag = flag
        self.step_number = step_number
        self.object_name = object_name
        self.data_filepath = data_filepath
        self.all_filelist = self.get_all_filelist()

    def get_all_filelist(self):
        filelist = []
        obj_filepath = os.path.join(self.data_filepath, self.object_name)
        # get the video ids based on training or testing data
        with open(os.path.join('./datainfo', self.object_name, f'data_split_dict_{self.seed}.json'), 'r') as file:
            seq_dict = json.load(file)

        vid_list = seq_dict[self.flag]
        # go through all the selected videos and get the triplets: input(t, t+1), output(t+2)
        
        length = len(os.listdir(os.path.join(obj_filepath, str(1))))
        for vid_idx in vid_list:
            seq_filepath = os.path.join(obj_filepath, str(vid_idx))

            num_frames = len(os.listdir(seq_filepath))
            if num_frames!=length:
                print(num_frames,length)
                print(os.listdir( os.path.join(obj_filepath, str(0))),0)
                print(os.listdir(seq_filepath),vid_idx)
                # print(os.listdir(seq_filepath),0)
                raise ValueError("wrong data")
            suf = os.listdir(seq_filepath)[0].split('.')[-1]
            for p_frame in range(num_frames - self.step_number):
                par_list = []
                for p in range(self.step_number+1):
                    par_list.append(os.path.join(seq_filepath, str(p_frame + p) + '.' + suf))
                filelist.append(par_list)
        return filelist

    def __len__(self):
        return len(self.all_filelist)

    # 0, 1 -> 2, 3 
    def __getitem__(self, idx):
        par_list = self.all_filelist[idx]
        data = []
        for i in range(2):
            data.append(self.get_data(par_list[i])) # 0, 1
        data = torch.cat(data, 2)
        target = []
        target.append(self.get_data(par_list[-2])) # 2
        target.append(self.get_data(par_list[-1])) # 3
        target = torch.cat(target, 2)
        filepath = '_'.join(par_list[0].split('/')[-2:])
        return data, target#, filepath        
 
    def get_data(self, filepath):
        data = Image.open(filepath)
        data = data.resize((128, 128))
        data = np.array(data)
        data = torch.tensor(data)
        data = data.permute(2, 0, 1)
        return data