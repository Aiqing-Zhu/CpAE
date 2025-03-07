import torch

from .module import Module, DynNN

class PNN(Module):
    '''INN-based Poisson neural network.
    '''
    def __init__(self, inn, sympnet, recurrent=1):
        super(PNN, self).__init__()
        self.inn = inn
        self.sympnet = sympnet
        self.recurrent = recurrent
        
        self.dim = sympnet.dim
    
    def forward(self, x):
        x = self.inn(x)
        for i in range(self.recurrent):
            x = self.sympnet(x)
        return self.inn.inverse(x)
    
    def predict(self, x, steps=1, keepinitx=False, returnnp=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self.Dtype, device=self.Device)
        dim = x.size(-1)
        size = len(x.size())
        pred = [self.inn(x)]
        for _ in range(steps):
            pred.append(self.sympnet(pred[-1]))
        pred = list(map(self.inn.inverse, pred))
        if keepinitx:
            steps = steps + 1
        else:
            pred = pred[1:]
        res = torch.cat(pred, dim=-1)
        if steps > 1:
            res = res.view([-1, steps, dim][2 - size:])
        return res.cpu().detach().numpy() if returnnp else res
    
class AEPNN(DynNN):
    '''Autoencoder-based Poisson neural network.
    '''
    def __init__(self, ae, sympnet, lam=1, recurrent=1):
        super(AEPNN, self).__init__()
        self.ae = ae
        self.sympnet = sympnet
        self.lam = lam
        self.recurrent = recurrent
    
    def criterion(self, X, y):
        X_latent, y_latent = self.ae.encode(X), self.ae.encode(y)
        X_latent_step = X_latent
        for i in range(self.recurrent):
            X_latent_step = self.sympnet(X_latent_step)
        symp_loss = torch.nn.MSELoss()(X_latent_step, y_latent)
        ae_loss = torch.nn.MSELoss()(self.ae.decode(X_latent), X) + torch.nn.MSELoss()(self.ae.decode(y_latent), y)
        return symp_loss + self.lam * ae_loss
    
    def predict(self, x, steps=1, keepinitx=False, returnnp=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self.Dtype, device=self.Device)
        dim = x.size(-1)
        size = len(x.size())
        pred = [self.ae.encode(x)]
        for _ in range(steps):
            pred.append(self.sympnet(pred[-1]))
        pred = list(map(self.ae.decode, pred))
        if keepinitx:
            steps = steps + 1
        else:
            pred = pred[1:]
        res = torch.cat(pred, dim=-1)
        if steps > 1:
            res = res.view([-1, steps, dim][2 - size:])
        return res.cpu().detach().numpy() if returnnp else res
    
class AEODEN(DynNN):
    '''Autoencoder-based ODEnets.
    '''
    def __init__(self, ae, fnn, h=0.1, lam=1, recurrent=1):
        super(AEODEN, self).__init__()
        self.ae = ae
        self.oden_vf = fnn
        self.h=h
        self.lam = lam
        self.recurrent = recurrent
    
    def ODEN(self, x):
        return x+ self.h*self.oden_vf(x)
    
    def criterion(self, X, y):
        X_latent, y_latent = self.ae.encode(X), self.ae.encode(y)
        X_latent_step = X_latent
        for i in range(self.recurrent):
            X_latent_step = self.ODEN(X_latent_step)
        ode_loss = torch.nn.MSELoss()(X_latent_step, y_latent)
        ae_loss = torch.nn.MSELoss()(self.ae.decode(X_latent), X) + torch.nn.MSELoss()(self.ae.decode(y_latent), y)
        return ode_loss + self.lam * ae_loss
    
    def predict(self, x, steps=1, keepinitx=False, returnnp=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self.Dtype, device=self.Device)
        dim = x.size(-1)
        size = len(x.size())
        pred = [self.ae.encode(x)]
        for _ in range(steps):
            pred.append(self.ODEN(pred[-1]))
        pred = list(map(self.ae.decode, pred))
        if keepinitx:
            steps = steps + 1
        else:
            pred = pred[1:]
        res = torch.cat(pred, dim=-1)
        if steps > 1:
            res = res.view([-1, steps, dim][2 - size:])
        return res.cpu().detach().numpy() if returnnp else res