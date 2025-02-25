import torch

from .module import Module
from .fnn import FNN
import numpy as np

class InteNN(Module):
    '''encoder.
    '''
    def __init__(self, dim=11):
        super(InteNN, self).__init__()
        if dim % 2 == 0:
            dim =dim - 1 
        self.dim = dim  
        
        self.modus = self.__init_modules()
    
    def forward(self, x): 
        rbf0 = self.modus['rbf0'](x[..., 0:1,:, :])
        rbf1 = self.modus['rbf1'](x[..., 1:2,:, :])
        rbf2 = self.modus['rbf2'](x[..., 2:3,:, :])
        return torch.cat((rbf0, rbf1, rbf2), dim=-3)

    
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        size_x,size_y = self.dim, self.dim
        A = np.ones(size_y)[:, None] * np.arange(size_x) / size_x
        B = np.arange(size_y)[::-1, None] * np.ones(size_x) / size_x
        mu1, mu2=0.5, 0.5
        sigma=0.3
        custom = np.exp(- ((A - mu1) ** 2 + (B - mu2) ** 2)/2/ sigma**2)/2.5066/sigma/size_x/size_y
        for i in range(3):
            modules['rbf{}'.format(i)] =torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(self.dim, self.dim), stride=1, padding=int((self.dim-1)/2))
            modules['rbf{}'.format(i)].weight.data[0,0] = torch.tensor(custom)
            for para in modules['rbf{}'.format(i)].parameters():
                para.requires_grad = False
        
        return modules
 
    
class InteTranNN(Module):
    '''decoder.
    '''
    def __init__(self, dim=11):
        super(InteTranNN, self).__init__()
        if dim % 2 == 0:
            dim =dim - 1 
        self.dim = dim  
        
        self.modus = self.__init_modules()
    
    def forward(self, x): 
        rbf0 = self.modus['rbf0'](x[..., 0:1,:, :])
        rbf1 = self.modus['rbf1'](x[..., 1:2,:, :])
        rbf2 = self.modus['rbf2'](x[..., 2:3,:, :])
        return torch.cat((rbf0, rbf1, rbf2), dim=-3)

    
    def __init_modules(self):
        modules = torch.nn.ModuleDict() 
        for i in range(3):
            modules['rbf{}'.format(i)] =torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(self.dim, self.dim), stride=1, padding=int((self.dim-1)/2))

        
        return modules

