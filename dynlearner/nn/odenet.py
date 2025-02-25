import torch

from .module import DynNN
from .fnn import FNN
from ..integrator.rungekutta import RK4, Integrator_list 

from .sympnet import LASympNet
from .vpnet import LAVPNet

# class ODENet(DynNN):
#     '''Neural ODEs.
#     '''
#     def __init__(self, dim=4, layers=2, width=128, activation='tanh', initializer='orthogonal', 
#                  integrator='euler', steps=1, iterations =1):
#         super(ODENet, self).__init__()
#         self.dim = dim
#         self.layers = layers
#         self.width = width       
#         self.activation = activation
#         self.initializer = initializer
#         self.integrator = integrator
#         self.steps = steps
#         self.iterations = iterations
        
#         self.modus = self.__init_modules()
    
#     def criterion(self, x0h, x1):
#         x0, h = (x0h[..., :-1], x0h[..., -1:])
#         return self.integrator_loss(x0, x1, h)
    
#     def predict(self, x0, h=0.1, steps=1, keepinitx=False, returnnp=False):
#         solver = RK4(self.vf, N= int(h/0.001)) 
#         res = solver.flow(x0, h, steps) if keepinitx else solver.flow(x0, h, steps)[..., 1:, :].squeeze()
#         return res.cpu().detach().numpy() if returnnp else res
        

#     def vf(self, x):
#         return self.modus['f'](x)
        
#     def __init_modules(self):
#         modules = torch.nn.ModuleDict()
#         modules['f'] = FNN(self.dim, self.dim, self.layers, self.width, self.activation, self.initializer)
#         return modules 
    
#     def integrator_loss(self, x0, x1, h):
#         n=int(self.steps)
#         solver = Integrator_list[self.integrator](self.vf, n)
#         x=solver.solve(x0, h)
#         return torch.nn.MSELoss()(x1, x)


import torch.nn as nn
from .module import Module
class ResNet(Module):
    '''ResNet.
    '''
    def __init__(self, ind=2, outd=2, layers=2, width=50, activation='relu', initializer='default', softmax=False):
        super(ResNet, self).__init__()
        self.ind = ind
        self.outd = outd
        self.layers = layers
        self.width = width
        self.activation = activation
        self.initializer = initializer
        self.softmax = softmax
        
        self.modus = self.__init_modules()
        self.__initialize()
        
    def forward(self, x):
        LinM = self.modus['LinM1'] 
        x=self.act(LinM(x))
        for i in range(2, self.layers):
            LinM = self.modus['LinM{}'.format(i)] 
            x=self.act(LinM(x)) + x
        x = self.modus['LinMout'](x)
        if self.softmax:
            x = nn.functional.softmax(x, dim=-1)
        return x
    
    def __init_modules(self):
        modules = nn.ModuleDict()
        if self.layers > 1:
            modules['LinM1'] = nn.Linear(self.ind, self.width)
            # modules['NonM1'] = self.Act
            for i in range(2, self.layers):
                modules['LinM{}'.format(i)] = nn.Linear(self.width, self.width)
                # modules['NonM{}'.format(i)] = self.Act
            modules['LinMout'] = nn.Linear(self.width, self.outd)
        else:
            modules['LinMout'] = nn.Linear(self.ind, self.outd)
            
        return modules
    
    def __initialize(self):
        for i in range(1, self.layers):
            self.weight_init_(self.modus['LinM{}'.format(i)].weight)
            nn.init.constant_(self.modus['LinM{}'.format(i)].bias, 0)
        self.weight_init_(self.modus['LinMout'].weight)
        nn.init.constant_(self.modus['LinMout'].bias, 0)
    
class ODENet(DynNN):
    '''Neural ODEs.
    '''
    def __init__(self, dim=4, h=0.1, layers=2, width=128, activation='tanh', initializer='orthogonal', 
                 integrator='explicit midpoint', steps=1):
        super(ODENet, self).__init__()
        self.dim = dim
        self.h=h
        self.layers = layers
        self.width = width       
        self.activation = activation
        self.initializer = initializer
        self.integrator = integrator
        self.steps = steps        
        self.modus = self.__init_modules()
    
    def criterion(self, x0, x1):
        return self.integrator_loss(x0, x1, self.h)
    
    def predict(self, x0, h=0.1, steps=1, keepinitx=False, returnnp=False):
        # solver = RK4(self.vf, N= int(h/0.001)) 
        n=int(self.steps)
        solver = Integrator_list[self.integrator](self.vf, n)
        res = solver.flow(x0, h, steps) if keepinitx else solver.flow(x0, h, steps)[..., 1:, :].squeeze()
        return res.cpu().detach().numpy() if returnnp else res
    
    def ODE_forward(self, x0):
        n=int(self.steps)
        solver = Integrator_list[self.integrator](self.vf, n)
        x=solver.solve(x0, self.h)
        return x

    def vf(self, x):
        return self.modus['f'](x)
        
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        modules['f'] = FNN(self.dim, self.dim, self.layers, self.width, self.activation, self.initializer)
        return modules 
    
    def integrator_loss(self, x0, x1, h):
        n=int(self.steps)
        solver = Integrator_list[self.integrator](self.vf, n)
        x=solver.solve(x0, h)
        loss=torch.nn.MSELoss()(x1[0], x)/h**2
        for i in range(1, x1.shape[0]):
            x=solver.solve(x, h)
            loss = loss + torch.nn.MSELoss()(x1[i], x)/h**2
        return loss
    
from .InteNN import InteNN

class AE_ODENet(DynNN):
    '''Neural ODEs based on autoencoder.
    '''
    def __init__(self, ae, lam=1, latent_dyn='ode', Inte=False,
                 h=0.1, dim=4, layers=2, width=128, activation='tanh', initializer='orthogonal', 
                 integrator='explicit midpoint', steps=1,
                 
                ):
        super(AE_ODENet, self).__init__()
        self.ae=ae
        self.lam = lam
        self.h=h
        self.Inte = Inte
        self.latent_dyn=latent_dyn
        
        self.dim = dim
        self.layers = layers
        self.width = width       
        self.activation = activation
        self.initializer = initializer
        self.integrator = integrator
        self.steps = steps 
        
        self.modus = self.__init_modules()
        
        symp_LAlayers = 3
        symp_LAsublayers = 2 
        symp_activation = 'sigmoid'
        self.sympnet = LASympNet(dim, symp_LAlayers, symp_LAsublayers, symp_activation)

        vp_LAlayers = 3
        vp_LAsublayers = 2 
        vp_activation = 'sigmoid'
        self.vpnet = LAVPNet(dim, vp_LAlayers, vp_LAsublayers, vp_activation)
        
        if self.Inte:
            self.InteNN=InteNN()

    def ODE_forward(self, x0, h):
        if self.latent_dyn=='ode':
            n=int(self.steps)
            solver = Integrator_list[self.integrator](self.vf, n)
            x0 = x0.flatten(start_dim=1)
            x=solver.solve(x0, h)
            x=x.unsqueeze(-1).unsqueeze(-1)
        if self.latent_dyn=='dis':
            x=x
        if self.latent_dyn=='symp':
            x0 = x0.flatten(start_dim=1)
            x=self.sympnet(x0)
            x=x.unsqueeze(-1).unsqueeze(-1)
            
        if self.latent_dyn=='vp':
            # x0 = x0.flatten(start_dim=1)
            x=self.vpnet(x0)
            # x=x.unsqueeze(-1).unsqueeze(-1)
        return x            
            
            
    def criterion0(self, X, y):
        if self.Inte:
            X_I = self.InteNN(X)
            y_I = self.InteNN(y)
            X_latent, y_latent = self.ae.encoder(X_I), self.ae.encoder(y_I)
        else:
            X_latent, y_latent = self.ae.encoder(X), self.ae.encoder(y)           

        y_latent_pre = self.ODE_forward(X_latent, self.h)
        
        # ode_loss = torch.nn.MSELoss()(y_latent_pre, y_latent)
        y_pre = self.ae.decoder(y_latent_pre)
        recon_loss = torch.nn.MSELoss()(self.ae.decoder(X_latent), X) + torch.nn.MSELoss()(y_pre, y)
        
        # ode_loss = torch.nn.MSELoss()(self.ae.encoder(y_pre), y_latent_pre)
        
        return recon_loss #+ self.lam * ode_loss            

    def criterion(self, X, y):
        if self.Inte:
            X_I = self.InteNN(X)
            y_I = self.InteNN(y)
            X_latent, y_latent = self.ae.encoder(X_I), self.ae.encoder(y_I)
        else:
            X_latent, y_latent = self.ae.encoder(X), self.ae.encoder(y)           

        y_latent_pre = self.ODE_forward(X_latent, self.h)
        
        ode_loss = torch.nn.MSELoss()(y_latent_pre, y_latent)
        y_pre = self.ae.decoder(y_latent_pre)
        recon_loss = torch.nn.MSELoss()(self.ae.decoder(X_latent), X) + torch.nn.MSELoss()(y_pre, y)
        
        # ode_loss = torch.nn.MSELoss()(self.ae.encoder(y_pre), y_latent_pre)
        
        return recon_loss + self.lam * ode_loss       
    
    def criterion1(self, X, y):
        if self.Inte:
            X_I = self.InteNN(X) 
            X_latent = self.ae.encoder(X_I) 
        else:
            X_latent = self.ae.encoder(X)       
            
        y_pre = self.ae.decoder(X_latent)
        recon_loss = torch.nn.MSELoss()(y_pre, y) 
        return recon_loss 

    def test_criterion1(self, X, y):
        if self.Inte:
            X_I = self.InteNN(X) 
            X_latent = self.ae.encoder(X_I) 
        else:
            X_latent = self.ae.encoder(X)       
            
        y_pre = self.ae.decoder(X_latent)
        recon_loss = torch.nn.MSELoss()(y_pre, y) 
        return recon_loss    

    def predict_onestep1(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(x, dtype=self.dtype, device=self.device)
        X_latent = self.encoder(X) 
        y_pre = self.decoder(X_latent)
        return y_pre
    def predict_multistep1(self, x, steps=2, returnnp=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self.dtype, device=self.device)
        pred=[x]
        for _ in range(steps):
            pred.append(self.predict_onestep(pred[-1]))
        if returnnp:
            return list(map(self.ReturnNp, pred))
        else:
            return pred      
    
    
    def test_criterion(self, X, y):
        if self.Inte:
            X_I = self.InteNN(X)
            y_I = self.InteNN(y)
            X_latent, y_latent = self.ae.encoder(X_I), self.ae.encoder(y_I)
        else:
            X_latent, y_latent = self.ae.encoder(X), self.ae.encoder(y)           

        y_latent_pre = self.ODE_forward(X_latent, self.h)
        
        ode_loss = torch.nn.MSELoss()(y_latent_pre, y_latent)
        recon_loss1 = torch.nn.MSELoss()(self.ae.decoder(X_latent), X)
        recon_loss2 = torch.nn.MSELoss()(self.ae.decoder(y_latent_pre), y)
        recon_loss = recon_loss1 + recon_loss2
        print('ode_loss,  recon_loss:', ode_loss,  recon_loss1, recon_loss2)
        return recon_loss + self.lam * ode_loss           
 
    def predict_onestep(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self.dtype, device=self.device)
        X_latent = self.encoder(x)
        y_latent_pre = self.ODE_forward(X_latent, self.h)
        y_pre = self.decoder(y_latent_pre)
        return y_pre
    def predict_multistep(self, x, steps=2, returnnp=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self.dtype, device=self.device)
        pred=[x]
        for _ in range(steps):
            pred.append(self.predict_onestep(pred[-1]))
        if returnnp:
            return list(map(self.ReturnNp, pred))
        else:
            return pred    
        
    def predict(self, x, steps=2, returnnp=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self.dtype, device=self.device)
        X_latent = self.encoder(x) 
        latent = [X_latent]
        for _ in range(steps):
            latent.append(self.ODE_forward(latent[-1], self.h))
        pred = list(map(self.decoder, latent))
 
        if returnnp:
            return list(map(self.ReturnNp, pred)), list(map(self.ReturnNp, latent))
        else: 
            return pred, latent                       
            
        
        
        
        
#     def predict(self, x, h=0.1, steps=1, keepinitx=False, returnnp=False):
#         n=int(self.steps)
#         solver = Integrator_list[self.integrator](self.vf, n)

#         if not isinstance(x, torch.Tensor):
#             x = torch.tensor(x, dtype=self.dtype, device=self.device)

#         latent = [self.encoder(x)]
#         for _ in range(steps):
#             latent.append(solver.solve(latent[-1], h))
#         pred = list(map(self.decoder, latent))
#         if keepinitx:
#             steps = steps + 1
#         else:
#             pred = pred[1:]
#             latent=latent[1:]
#         if returnnp:
#             return list(map(self.ReturnNp, pred)), list(map(self.ReturnNp, latent))
#         else: 
#             return pred, latent
        
    def decoder(self, x):
        return self.ae.decoder(x)
    
    def encoder(self, x):
        if self.Inte:
            x = self.InteNN(x)
            
        return self.ae.encoder(x)
        
           
    def ReturnNp(self, tensor):
        return tensor.cpu().detach().numpy()

    def vf(self, x):
        return self.modus['f'](x)
        
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        modules['f'] = FNN(self.dim, self.dim, self.layers, self.width, self.activation, self.initializer)
        return modules 
    
    def integrator_loss(self, x0, x1, h):
        n=int(self.steps)
        solver = Integrator_list[self.integrator](self.vf, n)
        x=solver.solve(x0, h)
        return torch.nn.MSELoss()(x1, x)
    

       
        
class AE_Net(DynNN):
    '''autoencoder for comparison
    '''
    def __init__(self, ae):
        super(AE_ODENet, self).__init__()
        self.ae=ae
        if self.Inte:
            self.InteNN=InteNN()       

    
    def criterion(self, X, y):
        if self.Inte:
            X_I = self.InteNN(X) 
            X_latent = self.ae.encoder(X_I) 
        else:
            X_latent = self.ae.encoder(X)       
            
        y_pre = self.ae.decoder(X_latent)
        recon_loss = torch.nn.MSELoss()(y_pre, y) 
        return recon_loss 

    def test_criterion(self, X, y):
        if self.Inte:
            X_I = self.InteNN(X) 
            X_latent = self.ae.encoder(X_I) 
        else:
            X_latent = self.ae.encoder(X)       
            
        y_pre = self.ae.decoder(X_latent)
        recon_loss = torch.nn.MSELoss()(y_pre, y) 
        return recon_loss    

    def predict_onestep(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(x, dtype=self.dtype, device=self.device)
        X_latent = self.encoder(X) 
        y_pre = self.decoder(X_latent)
        return y_pre
    def predict_multistep(self, x, steps=2, returnnp=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self.dtype, device=self.device)
        pred=[x]
        for _ in range(steps):
            pred.append(self.predict_onestep(pred[-1]))
        if returnnp:
            return list(map(self.ReturnNp, pred))
        else:
            return pred      
    
    

    def decoder(self, x):
        return self.ae.decoder(x)
    
    def encoder(self, x):
        if self.Inte:
            x = self.InteNN(x)
            
        return self.ae.encoder(x)
        
           
    def ReturnNp(self, tensor):
        return tensor.cpu().detach().numpy()

    
    
class latent_ODENet(DynNN):
    '''Neural ODEs based on autoencoder.
    '''
    def __init__(self, ae, Inte=False,
                 h=0.1, dim=4, layers=2, width=128, activation='tanh', initializer='orthogonal', 
                 integrator='explicit midpoint', steps=1,
                 
                ):
        super(latent_ODENet, self).__init__()
        self.ae=ae.eval()
        for param in ae.parameters():
            param.requires_grad = False
        
        self.Inte=Inte
        
        
        self.h=h  
        self.dim = dim
        self.layers = layers
        self.width = width       
        self.activation = activation
        self.initializer = initializer
        self.integrator = integrator
        self.steps = steps 
        
        self.modus = self.__init_modules()
        if self.Inte:
            self.InteNN=InteNN()
  
    def criterion(self, X, y):
        if self.Inte:
            X_I = self.InteNN(X)
            y_I = self.InteNN(y)
            X_latent, y_latent = self.ae.encoder(X_I), self.ae.encoder(y_I)
        else:
            X_latent, y_latent = self.ae.encoder(X), self.ae.encoder(y)                 
        return self.integrator_loss(X_latent.flatten(start_dim=1), y_latent.flatten(start_dim=1), self.h)

    
    def test_criterion(self, X, y):
        if self.Inte:
            X_I = self.InteNN(X)
            y_I = self.InteNN(y)
            X_latent, y_latent = self.ae.encoder(X_I), self.ae.encoder(y_I)
        else:
            X_latent, y_latent = self.ae.encoder(X), self.ae.encoder(y)           
        
        ode_loss = self.integrator_loss(X_latent.flatten(start_dim=1), y_latent.flatten(start_dim=1), self.h)
        recon_loss1 = torch.nn.MSELoss()(self.ae.decoder(X_latent), X)
        recon_loss2 = torch.nn.MSELoss()(self.ae.decoder(y_latent), y) 
        print('ode_loss,  recon_loss:', ode_loss,  recon_loss1, recon_loss2)
        return ode_loss           
  
        
    def predict(self, x, steps=2, returnnp=False):
        n=int(self.steps)
        solver = Integrator_list[self.integrator](self.vf, n)        
        
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self.dtype, device=self.device)
        X_latent = self.encoder(x) 
        latent = [X_latent]
        for _ in range(steps):
            latent.append(solver.solve(latent[-1], h))
        pred = list(map(self.decoder, latent))
 
        if returnnp:
            return list(map(self.ReturnNp, pred)), list(map(self.ReturnNp, latent))
        else: 
            return pred, latent                       
            
        
        
        
        
#     def predict(self, x, h=0.1, steps=1, keepinitx=False, returnnp=False):
#         n=int(self.steps)
#         solver = Integrator_list[self.integrator](self.vf, n)

#         if not isinstance(x, torch.Tensor):
#             x = torch.tensor(x, dtype=self.dtype, device=self.device)

#         latent = [self.encoder(x)]
#         for _ in range(steps):
#             latent.append(solver.solve(latent[-1], h))
#         pred = list(map(self.decoder, latent))
#         if keepinitx:
#             steps = steps + 1
#         else:
#             pred = pred[1:]
#             latent=latent[1:]
#         if returnnp:
#             return list(map(self.ReturnNp, pred)), list(map(self.ReturnNp, latent))
#         else: 
#             return pred, latent
        
    def decoder(self, x):
        return self.ae.decoder(x)
    
    def encoder(self, x):
        if self.Inte:
            x = self.InteNN(x)
            
        return self.ae.encoder(x)
        
           
    def ReturnNp(self, tensor):
        return tensor.cpu().detach().numpy()

    def vf(self, x):
        return self.modus['f'](x)
        
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        modules['f'] = FNN(self.dim, self.dim, self.layers, self.width, self.activation, self.initializer)
        return modules 
    
    def integrator_loss(self, x0, x1, h):
        n=int(self.steps)
        solver = Integrator_list[self.integrator](self.vf, n)
        x=solver.solve(x0, h)
        return torch.nn.MSELoss()(x1, x)