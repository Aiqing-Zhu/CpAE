import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import dynlearner as ln
from dynlearner.integrator.hamiltonian import SV
import matplotlib.pyplot as plt


class CirData(ln.Data):
    '''Images.
    '''
    def __init__(self, h, train_num, test_num, size=50):
        super(CirData, self).__init__()
        self.h = h
        self.train_num = train_num
        self.test_num = test_num 
        
        self.x0 = np.array([0, 0.35])
        self.r = 0.1
        self.size = size
        self.solver = SV(None, self.dH, iterations=1, order=4, N=max(int(self.h * 1e3), 1))
        self.__init_data()
        
    def dH(self, p, q):
        q1 = q[..., :2]
        q2 = q[..., 2:]
        r = np.linalg.norm(q1 - q2, axis=-1, keepdims=True)
        dp = p
        dq = (r ** -3) * np.hstack((q1 - q2, q2 - q1))
        return dp, dq

    def show_image(self, flat_data, save_path=None):
        plt.figure()
        plt.imshow(flat_data.reshape(2 * self.size, 2 * self.size), cmap='gray')
        plt.axis('off')
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        plt.close()

    def image(self, x):
        size_x = 1 * self.size
        size_y = 1 * self.size
        A = np.ones(size_y)[:, None] * np.arange(size_x) / self.size-0.5
        B = np.arange(size_y)[::-1, None] * np.ones(size_x) / self.size -0.5
        sigmoid = lambda x: 1/(1+np.exp(-x))
        b1 = sigmoid((self.r ** 2 - ((A - x[0]) ** 2 + (B - x[1]) ** 2)) * 500)
        b2 = sigmoid((self.r ** 2 - ((A - x[0]) ** 2 + (B - x[1]) ** 2)) * 500)
        return np.expand_dims(np.maximum(b1, b2), axis=0)
    def cir(self, x, h, num):
        A=np.array([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]])
        flow=[x]
        for i in range(num):
            flow.append(A@flow[-1])
        return np.array(flow)
        
        
    def __init_data(self): 
        
        flow_train = self.cir(self.x0, self.h, self.train_num)
        flow_test = self.cir(flow_train[-1], self.h, self.test_num)        
        
        flow_train_image = np.array(list(map(self.image, flow_train))) 
        flow_test_image = np.array(list(map(self.image, flow_test))) 
        self.X_train_raw, self.y_train_raw = flow_train[:-1], flow_train[1:]
        self.X_test_raw, self.y_test_raw = flow_test[:-1], flow_test[1:]
        self.X_train, self.y_train = flow_train_image[:-1], flow_train_image[1:]
        self.X_test, self.y_test = flow_test_image[:-1], flow_test_image[1:]

def main(re_lam=1, seed=0, gpu_index=2, size=50):
    device = 'gpu' # 'cpu' or 'gpu'
    torch.cuda.set_device(gpu_index)
    # data
    h = 0.1
    train_num = 70
    test_num = 70


    # training
    lr = 0.001
    iterations = 100000
    print_every = 1000

    data = CirData(h, train_num, test_num, size=size)

    net = CAE_hyp(in_channels=1, lam=1, latent_dyn='vp', re_lam=re_lam, large=True, size=size)

    args = {
        'filename': 'Cir_ablation_re{}seed{}size{}'.format(re_lam, seed, size),
        'data': data,
        'net': net,
        'criterion': None,
        'optimizer': 'adam',
        'lr': lr,
        'iterations': iterations,
        'batch_size': None,
        'print_every': print_every,
        'save': True,
        'callback': None,
        'dtype': 'float',
        'device': device
    }

    ln.Brain.Init(**args)
    ln.Brain.Run()
    ln.Brain.Restore()
    ln.Brain.Output()

    
class LatentData(ln.Data):
    def __init__(self, net, data):
        super(LatentData, self).__init__()
        self.net=net
        self.data=data
 
        self.__init_data()
        
    @property
    def dim(self):
        return 2
 
    
    def __init_data(self):       
        self.train_traj =self.net.encoder(torch.tensor(self.data.X_train, dtype=torch.float32)).detach()
        self.test_traj  =self.net.encoder(torch.tensor(self.data.X_test, dtype=torch.float32)).detach()
        self.X_train = self.train_traj[:-1]
        self.y_train = self.train_traj[1:].unsqueeze(0)
        self.X_test = self.test_traj[:-1]
        self.y_test = self.test_traj[1:].unsqueeze(0)
def train_cir_latent_dynamics(re_lam=1, seed=2, size=48):
    device = 'cpu' # 'cpu' or 'gpu'
    
    torch.cuda.set_device(0)

    AE = torch.load('outputs/Cir_ablation_re{}seed{}size{}/model_best.pkl'.format(re_lam, seed, size), map_location='cpu')
    ImgData = CirData(h=0.1, train_num=70, test_num=50, size=size)
    data = LatentData(net=AE, data=ImgData)
    print(data.X_train.shape, data.y_train.shape)
    print(data.X_test.shape, data.y_test.shape)

    filename = 'Cir_ablation_latent_re{}seed{}size{}'.format(re_lam, seed, size)
    
    net = ln.nn.ODENet(dim=2, h=0.1, layers=2, width=64, activation='tanh', steps=2)


    print(filename)
    args = {
        'filename': filename,
        'data': data,
        'net': net,
        'criterion': None,
        'optimizer': 'adam',
        'lr': 0.01,
        'lr_decay': 10,
        'iterations': 50000,
        'batch_size': None,
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

def conv2d_bn_sigmoid(inch,outch,kernel_size,stride=1,padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.Conv2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.Sigmoid()
    )
    return convlayer

def deconv_sigmoid(inch,outch,kernel_size,stride=1,padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.Sigmoid()
    )
    return convlayer
       
class EncoderDecoder_largekernal(torch.nn.Module):
    def __init__(self, in_channels=3, channels1=8, size=50, large=True):
        super(EncoderDecoder_largekernal,self).__init__()
        if large:
            self.conv_stack1 = torch.nn.Sequential(
                conv2d_bn_sigmoid(in_channels,32,(1*size,size), stride=2, padding=0),
            )
            self.ldeconv_1 = deconv_sigmoid(32,1,(1*size,size),stride=2, padding=0)

        self.regu_modules=[self.conv_stack1]


    def encoder(self, x):
        conv1_out = self.conv_stack1(x)
        return conv1_out
    def decoder(self, x):
        deconv1_out = self.ldeconv_1(x)
        return deconv1_out       
     

    def forward(self,x, reconstructed_latent=None, refine_latent=False):
        if refine_latent != True:
            latent = self.encoder(x)
            out = self.decoder(latent)
            return out, latent
        else:
            latent = self.encoder(x)
            out = self.decoder(reconstructed_latent)
            return out, latent        
        
def ijkl2(array):
    array0 = array.unsqueeze(-3).unsqueeze(-3)
    array1 = array.unsqueeze(-1).unsqueeze(-1)
    return (array0-array1)**2
pad = torch.nn.ZeroPad2d(padding=(1, 1,1,1))

def f(x):
    return torch.exp(-x/10)
    
def continuous_regu(weight, f=f):
    aug_weight = pad(weight)
 
    xx     = np.linspace(0,1,weight.shape[-2]+2)
    yy     = np.linspace(0,1,weight.shape[-1]+2)
    XX, YY = np.meshgrid(yy,xx)

    XX, YY = torch.tensor(XX, dtype=weight.dtype, device=weight.device), torch.tensor(YY, dtype=weight.dtype, device=weight.device)
    
    regu = ijkl2(aug_weight)* f(ijkl2(XX) + ijkl2(YY))
    return regu.mean()*2*10



################
def Iden(x):
    return x

class CAE_hyp(ln.nn.DynNN):
    '''Adding Constraints to Latent Dynamics of the Autoencoder.
    '''
    def __init__(self, in_channels=1, latent_dim=2, lam=1, latent_dyn='vp', re_lam=1, large=True, Inte=True, size=50, 
                 h=0.1, layers=2, width=128, activation='tanh', initializer='orthogonal', 
                 integrator='explicit midpoint', steps=1,
                 
                ):
        super(CAE_hyp, self).__init__()
        self.re_lam=re_lam
        self.latent_dim= latent_dim
        self.ae=EncoderDecoder_largekernal(in_channels, large=large, size=size)
        self.encoder_output = nn.Linear(32, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 32)

        self.lam = lam
        self.latent_dyn=latent_dyn 
    
    def regularization(self, x=None, y=None):
        
        if self.re_lam==0:
            return 0
        elif self.re_lam==-1:
            con_loss = []
            for conv in self.ae.regu_modules:
                for module in conv.modules():
                    if type(module) is torch.nn.Conv2d and module.weight.shape[-1]>10:
                        con_loss.append( ((module.weight)**2).mean())
            return sum(con_loss) 
        else:
            con_loss = []
            for conv in self.ae.regu_modules:
                for module in conv.modules():
                    if type(module) is torch.nn.Conv2d and module.weight.shape[-1]>10:
                        con_loss.append(continuous_regu(module.weight))

            return sum(con_loss)*self.re_lam
                

    def encoder(self, x):
        cnn_out = self.ae.encoder(x).flatten(start_dim=1)
        y=self.encoder_output(cnn_out)
        return y
    
    def decoder(self, x):
        x= self.decoder_input(x)
        x = self.ae.decoder(x.unsqueeze(-1).unsqueeze(-1))
        return x       
                   

    def criterion(self, X, y):
        recon_loss = torch.nn.MSELoss()(self.decoder(self.encoder(X)), X) 
        return recon_loss      

    def predict(self, x, steps=2, returnnp=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self.dtype, device=self.device)
        X_latent = self.encoder(x) 
        latent = [X_latent]
        for _ in range(steps):
            latent.append(self.latent_forward(latent[-1]))
        pred = list(map(self.decoder, latent))
 
        if returnnp:
            return list(map(self.ReturnNp, pred)), np.array(list(map(self.ReturnNp, latent))).squeeze()
        else: 
            return pred, latent     
 
    
if __name__ == '__main__':
    seed=0
    gpu_index=1

    main(re_lam=-1, seed=seed,  gpu_index=gpu_index, size=48)
    main(re_lam=0, seed=seed,  gpu_index=gpu_index, size=48)
    main(re_lam=1, seed=seed,  gpu_index=gpu_index, size=48)
    train_cir_latent_dynamics(re_lam=-1, seed=seed, size=48)
    train_cir_latent_dynamics(re_lam=0, seed=seed, size=48)
    train_cir_latent_dynamics(re_lam=1, seed=seed, size=48)