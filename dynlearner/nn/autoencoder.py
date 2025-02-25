import torch

from .module import Module
from .fnn import FNN
from .InteNN import InteNN, InteTranNN 


class AE(Module):
    '''Autoencoder.
    '''
    def __init__(self, data_dim, latent_dim, depth, width, activation='sigmoid', initializer='default'):
        super(AE, self).__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.depth = depth
        self.width = width
        self.activation = activation
        self.initializer = initializer
        
        self.modus = self.__init_modules()
    
    def forward(self, x):
        return self.modus['decoder'](self.modus['encoder'](x))
    
    def encode(self, x, returnnp=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self.dtype, device=self.device)
        return self.modus['encoder'](x).cpu().detach().numpy() if returnnp else self.modus['encoder'](x)
    
    def decode(self, x, returnnp=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self.dtype, device=self.device)
        return self.modus['decoder'](x).cpu().detach().numpy() if returnnp else self.modus['decoder'](x)
    
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        modules['encoder'] = FNN(self.data_dim, self.latent_dim, self.depth, self.width, self.activation, self.initializer)
        modules['decoder'] = FNN(self.latent_dim, self.data_dim, self.depth, self.width, self.activation, self.initializer)         
        return modules
    
class Inte_AE(Module):
    '''Autoencoder based on integration CNN.
    '''
    def __init__(self, size=11):
        super(Inte_AE, self).__init__()
        self.size = size
        self.modus = self.__init_modules()
    
    def forward(self, x):
        return self.modus['InteTranNN'](self.modus['InteNN'](x))
    
    def encoder(self, x, returnnp=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self.dtype, device=self.device)
        return self.modus['InteNN'](x).cpu().detach().numpy() if returnnp else self.modus['InteNN'](x)
    
    def decoder(self, x, returnnp=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self.dtype, device=self.device)
        return self.modus['InteTranNN'](x).cpu().detach().numpy() if returnnp else self.modus['InteTranNN'](x)
    
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        modules['InteNN'] = InteNN(self.size)
        modules['InteTranNN'] = InteTranNN(self.size)
        return modules