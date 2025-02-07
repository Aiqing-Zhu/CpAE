import numpy as np
import torch
import os
import json

import dynlearner as ln

class LatentData(ln.Data):
    def __init__(self, step_number=1, skip=1, seed=1, 
                 object_name='elastic_pendulum', dimension=6, suffix='_caehyp_vp0'):
        super(LatentData, self).__init__() 
        
       
        self.seed = seed 
        self.step_number = step_number
        self.skip=skip
        self.object_name = object_name
        self.suffix=suffix 
        self.dimension=dimension
        
        self.train_vid_list, self.test_vid_list=None, None
        self.__init_data()
        
    @property
    def dim(self):
        return self.dimension
 
    
    def __init_data(self):       
        self.train_vid_list, self.X_train, self.y_train = self.get_all_data(flag='train')
        self.test_vid_list, self.X_test, self.y_test = self.get_all_data(flag='test')
        self.val_vid_list, self.X_val, self.y_val = self.get_all_data(flag='val')

    def get_all_data(self, flag):
        filelist = []
        obj_filepath = 'training_data/latent_'+self.object_name
        with open(os.path.join('./datainfo', self.object_name, f'data_split_dict_{self.seed}.json'), 'r') as file:
            seq_dict = json.load(file)
        vid_list = seq_dict[flag]
        
        data_X=[]
        data_y=[]
        for vid_idx in vid_list:
            seq_filepath = os.path.join(obj_filepath + self.suffix, str(vid_idx))
            traj = np.load(seq_filepath+'.npy')[::self.skip]
            data_X.append(traj[:-self.step_number])
            
            data_y_one=[]
            for i in range(1, self.step_number):
                data_y_one.append(traj[np.newaxis, i:-self.step_number+i])
            data_y_one.append(traj[np.newaxis, self.step_number:])
            data_y.append(np.concatenate(data_y_one, axis=0)) 
        return (vid_list, 
                torch.tensor(np.concatenate(data_X, axis=0), dtype=torch.float32), 
                torch.tensor(np.concatenate(data_y, axis=1), dtype=torch.float32)
               )
    


def train_latent_dynamics(dim=6, object_name='elastic_pendulum', latent_dyn='vp', gpu_index=0, seed=0):
    device = 'gpu' # 'cpu' or 'gpu'
    
    torch.cuda.set_device(gpu_index)

    
    data = LatentData(step_number=1, skip=1, object_name=object_name, suffix='_caehyp_'+latent_dyn+'{}'.format(seed))
    print(data.X_train.shape, data.y_train.shape)
    print(data.X_test.shape, data.y_test.shape)

    filename = 'latent_'+object_name + '_caehyp_' + latent_dyn+'{}'.format(seed)
    
    net = ln.nn.ODENet(dim=dim, h=0.01, layers=3, width=256, activation='tanh', steps=2)
    print(net.criterion(data.X_train, data.y_train))

    print(filename, 'use latent ode, 1step 1skip, latent ode, layer3 width256 tanh, iteration 50w， batch 10000 lr 0.001')
    args = {
        'filename': filename,
        'data': data,
        'net': net,
        'criterion': None,
        'optimizer': 'adam',
        'lr': 0.01,
        'lr_decay': 10,
        'iterations': 500000,
        'batch_size': 20000,
        'print_every': 1000,
        'save': True,
        'callback': None,
        'dtype': 'float',
        'device': device
    }
    

    ln.Brain.Init(**args)
    ln.Brain.Run()
    ln.Brain.Restore()
    ln.Brain.Output()
    

if __name__ == '__main__':
    train_latent_dynamics(object_name='elastic_pendulum', latent_dyn='vp', gpu_index=3, seed=0)
    train_latent_dynamics(object_name='elastic_pendulum', latent_dyn='vp', gpu_index=3, seed=1)
    train_latent_dynamics(object_name='elastic_pendulum', latent_dyn='vp', gpu_index=3, seed=2)