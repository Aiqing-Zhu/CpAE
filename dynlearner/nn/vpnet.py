import torch
import torch.nn as nn

from .module import Module

class LinearModule(Module):
    '''Linear volume-preserving module.
    '''
    def __init__(self, dim, layers):
        super(LinearModule, self).__init__()
        self.dim = dim
        self.layers = layers
        
        self.params = self.__init_params()
        
    def forward(self, pqh):
        p, q, h = pqh
        for i in range(self.layers):
            S = self.params['S{}'.format(i + 1)]
            if i % 2 == 0:
                p = p + q @ (S) * h
            else:
                q = p @ (S) * h + q
        return p + self.params['bp'] * h, q + self.params['bq'] * h
    
    def __init_params(self):
        '''Si is distributed N(0, 0.01), and b is set to zero.
        '''
        d = int(self.dim / 2)
        params = nn.ParameterDict()
        for i in range(self.layers):
            params['S{}'.format(i + 1)] = nn.Parameter((torch.randn([d, d]) * 0.01).requires_grad_(True))
        params['bp'] = nn.Parameter(torch.zeros([d]).requires_grad_(True))
        params['bq'] = nn.Parameter(torch.zeros([d]).requires_grad_(True))
        return params
        
class ActivationModule(Module):
    '''Activation volume-preserving module.
    '''
    def __init__(self, dim, activation, mode):
        super(ActivationModule, self).__init__()
        self.dim = dim
        self.activation = activation
        self.mode = mode
        
        self.params = self.__init_params()
        
    def forward(self, pqh):
        p, q, h = pqh
        if self.mode == 'up':
            return p + self.act(q) * self.params['a'] * h, q
        elif self.mode == 'low':
            return p, self.act(p) * self.params['a'] * h + q
        else:
            raise ValueError
            
    def __init_params(self):
        d = int(self.dim / 2)
        params = nn.ParameterDict()
        params['a'] = nn.Parameter((torch.randn([d]) * 0.01).requires_grad_(True))
        return params
            
class VPNet(Module):
    def __init__(self):
        super(VPNet, self).__init__()
        self.dim = None
        
    def predict(self, xh, steps=1, keepinitx=False, returnnp=False):
        if not isinstance(xh, torch.Tensor):
            xh = torch.tensor(xh, dtype=self.dtype, device=self.device)
        dim = xh.size(-1)
        size = len(xh.size())
        if dim == self.dim:
            pred = [xh]
            for _ in range(steps):
                pred.append(self(pred[-1]))
        else:
            x0, h = xh[..., :-1], xh[..., -1:] 
            pred = [x0]
            for _ in range(steps):
                pred.append(self(torch.cat([pred[-1], h], dim=-1)))
        if keepinitx:
            steps = steps + 1
        else:
            pred = pred[1:]
        res = torch.cat(pred, dim=-1).view([-1, steps, self.dim][2 - size:])
        return res.cpu().detach().numpy() if returnnp else res

class LAVPNet(VPNet):
    '''LA-VPNet.
    Input: [num, dim] or [num, dim + 1]
    Output: [num, dim]
    '''
    def __init__(self, dim, layers=3, sublayers=2, activation='sigmoid'):
        super(LAVPNet, self).__init__()
        self.dim = dim
        self.layers = layers
        self.sublayers = sublayers
        self.activation = activation
        
        self.modus = self.__init_modules()
        
    def forward(self, pqh):
        d = int(self.dim / 2)
        if pqh.size(-1) == self.dim + 1:
            p, q, h = pqh[..., :d], pqh[..., d:-1], pqh[..., -1:]
        elif pqh.size(-1) == self.dim:
            p, q, h = pqh[..., :d], pqh[..., d:], torch.ones_like(pqh[..., -1:])
        else:
            raise ValueError
        for i in range(self.layers - 1):
            LinM = self.modus['LinM{}'.format(i + 1)]
            ActM = self.modus['ActM{}'.format(i + 1)]
            p, q = ActM([*LinM([p, q, h]), h])
        return torch.cat(self.modus['LinMout']([p, q, h]), dim=-1)
    
    def __init_modules(self):
        modules = nn.ModuleDict()
        for i in range(self.layers - 1):
            modules['LinM{}'.format(i + 1)] = LinearModule(self.dim, self.sublayers)
            mode = 'up' if i % 2 == 0 else 'low'
            modules['ActM{}'.format(i + 1)] = ActivationModule(self.dim, self.activation, mode)
        modules['LinMout'] = LinearModule(self.dim, self.sublayers)
        return modules
            