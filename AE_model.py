import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def conv2d_bn_relu(inch,outch,kernel_size,stride=1,padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.Conv2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.ReLU()
    )
    return convlayer

def conv2d_bn(inch,outch,kernel_size,stride=1,padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.Conv2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.BatchNorm2d(outch)
    )
    return convlayer


def conv2d_bn_sigmoid(inch,outch,kernel_size,stride=1,padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.Conv2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.Sigmoid()
    )
    return convlayer

def deconv_sigmoid(inch,outch,kernel_size,stride=1,padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.Sigmoid()
    )
    return convlayer

def deconv_relu(inch,outch,kernel_size,stride=1,padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.ReLU()
    )
    return convlayer

class EncoderDecoder64x1x1(torch.nn.Module):
    def __init__(self, in_channels=3):
        super(EncoderDecoder64x1x1,self).__init__()

        self.conv_stack1 = torch.nn.Sequential(
            conv2d_bn_relu(in_channels,32,4,stride=2),
            conv2d_bn_relu(32,32,3)
        )
        self.conv_stack2 = torch.nn.Sequential(
            conv2d_bn_relu(32,32,4,stride=2),
            conv2d_bn_relu(32,32,3)
        )
        self.conv_stack3 = torch.nn.Sequential(
            conv2d_bn_relu(32,64,4,stride=2),
            conv2d_bn_relu(64,64,3)
        )
        self.conv_stack4 = torch.nn.Sequential(
            conv2d_bn_relu(64,64,4,stride=2),
            conv2d_bn_relu(64,64,3),
        )

        self.conv_stack5 = torch.nn.Sequential(
            conv2d_bn_relu(64,64,4,stride=2),
            conv2d_bn_relu(64,64,3),
        )

        self.conv_stack6 = torch.nn.Sequential(
            conv2d_bn_relu(64,64,4,stride=2),
            conv2d_bn_relu(64,64,3),
        )

        self.conv_stack7 = torch.nn.Sequential(
            conv2d_bn_relu(64,64,4,stride=2),
            conv2d_bn_relu(64,64,3),
        )

        self.conv_stack8 = torch.nn.Sequential(
            conv2d_bn_relu(64, 64, (3,4), stride=(1,2)),
            conv2d_bn_relu(64, 64, 3),
        )
        
        self.encoder_modules=[self.conv_stack1, 
                              self.conv_stack2,
                              self.conv_stack3,
                              self.conv_stack4,
                              self.conv_stack5,
                              self.conv_stack6]
        
        self.deconv_8 = deconv_relu(64,64,(3,4),stride=(1,2))
        self.deconv_7 = deconv_relu(67,64,4,stride=2)
        self.deconv_6 = deconv_relu(67,64,4,stride=2)
        self.deconv_5 = deconv_relu(67,64,4,stride=2)
        self.deconv_4 = deconv_relu(67,64,4,stride=2)
        self.deconv_3 = deconv_relu(67,32,4,stride=2)
        self.deconv_2 = deconv_relu(35,16,4,stride=2)
        self.deconv_1 = deconv_sigmoid(19,3,4,stride=2)

        self.predict_8 = torch.nn.Conv2d(64,3,3,stride=1,padding=1)
        self.predict_7 = torch.nn.Conv2d(67,3,3,stride=1,padding=1)
        self.predict_6 = torch.nn.Conv2d(67,3,3,stride=1,padding=1)
        self.predict_5 = torch.nn.Conv2d(67,3,3,stride=1,padding=1)
        self.predict_4 = torch.nn.Conv2d(67,3,3,stride=1,padding=1)
        self.predict_3 = torch.nn.Conv2d(67,3,3,stride=1,padding=1)
        self.predict_2 = torch.nn.Conv2d(35,3,3,stride=1,padding=1)

        self.up_sample_8 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,(3,4),stride=(1,2),padding=1,bias=False),
            torch.nn.Sigmoid()
        )

        self.up_sample_7 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )

        self.up_sample_6 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )

        self.up_sample_5 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )

        self.up_sample_4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )


    def encoder(self, x):
        conv1_out = self.conv_stack1(x)
        conv2_out = self.conv_stack2(conv1_out)
        conv3_out = self.conv_stack3(conv2_out)
        conv4_out = self.conv_stack4(conv3_out)
        conv5_out = self.conv_stack5(conv4_out)
        conv6_out = self.conv_stack6(conv5_out)
        conv7_out = self.conv_stack7(conv6_out)
        conv8_out = self.conv_stack8(conv7_out)
        return conv8_out

    def decoder(self, x):

        deconv8_out = self.deconv_8(x)
        predict_8_out = self.up_sample_8(self.predict_8(x))

        concat_7 = torch.cat([deconv8_out, predict_8_out], dim=1)
        deconv7_out = self.deconv_7(concat_7)
        predict_7_out = self.up_sample_7(self.predict_7(concat_7))

        concat_6 = torch.cat([deconv7_out,predict_7_out],dim=1)
        deconv6_out = self.deconv_6(concat_6)
        predict_6_out = self.up_sample_6(self.predict_6(concat_6))

        concat_5 = torch.cat([deconv6_out,predict_6_out],dim=1)
        deconv5_out = self.deconv_5(concat_5)
        predict_5_out = self.up_sample_5(self.predict_5(concat_5))

        concat_4 = torch.cat([deconv5_out,predict_5_out],dim=1)
        deconv4_out = self.deconv_4(concat_4)
        predict_4_out = self.up_sample_4(self.predict_4(concat_4))

        concat_3 = torch.cat([deconv4_out,predict_4_out],dim=1)
        deconv3_out = self.deconv_3(concat_3)
        predict_3_out = self.up_sample_3(self.predict_3(concat_3))

        concat2 = torch.cat([deconv3_out,predict_3_out],dim=1)
        deconv2_out = self.deconv_2(concat2)
        predict_2_out = self.up_sample_2(self.predict_2(concat2))

        concat1 = torch.cat([deconv2_out,predict_2_out],dim=1)
        predict_out = self.deconv_1(concat1)
        return predict_out
        

    def forward(self,x, reconstructed_latent, refine_latent):
        if refine_latent != True:
            latent = self.encoder(x)
            out = self.decoder(latent)
            return out, latent
        else:
            latent = self.encoder(x)
            out = self.decoder(reconstructed_latent)
            return out, latent

        
class EncoderDecoder_largekernal(torch.nn.Module):
    def __init__(self, in_channels=3, channels1=8):
        super(EncoderDecoder_largekernal,self).__init__()

        self.conv_stack1 = torch.nn.Sequential(
            conv2d_bn_relu(in_channels,16,12, stride=2, padding=5),
            conv2d_bn_relu(16,16,3)
        )
        self.conv_stack2 = torch.nn.Sequential(
            conv2d_bn_relu(16,32,12,stride=2, padding=5),
            conv2d_bn_relu(32,32,3)
        )
        self.conv_stack3 = torch.nn.Sequential(
            conv2d_bn_relu(32,64,12,stride=2, padding=5),
            conv2d_bn_relu(64,64,3)
        )
        self.conv_stack4 = torch.nn.Sequential(
            conv2d_bn_relu(64,64,4,stride=2),
            conv2d_bn_relu(64,64,3),
        )

        self.conv_stack5 = torch.nn.Sequential(
            conv2d_bn_relu(64,64,4,stride=2),
            conv2d_bn_relu(64,64,3),
        )

        self.conv_stack6 = torch.nn.Sequential(
            conv2d_bn_relu(64,64,4,stride=2),
            conv2d_bn_relu(64,64,3),
        )


        self.conv_stack7 = torch.nn.Sequential(
            conv2d_bn_relu(64,64,4,stride=2),
            conv2d_bn_relu(64,64,3),
        )

        self.conv_stack8 = torch.nn.Sequential(
            conv2d_bn_relu(64, 64, (3,4), stride=(1,2)),
            conv2d_bn_relu(64, 64, 3),
        )
        
        self.regu_modules=[self.conv_stack1, self.conv_stack2, self.conv_stack3]
        
        self.deconv_8 = deconv_relu(64,64,(3,4),stride=(1,2))
        self.deconv_7 = deconv_relu(67,64,4,stride=2)
        self.deconv_6 = deconv_relu(67,64,4,stride=2)
        self.deconv_5 = deconv_relu(67,64,4,stride=2)
        self.deconv_4 = deconv_relu(67,64,4,stride=2)
        self.deconv_3 = deconv_relu(67,32,4,stride=2)
        self.deconv_2 = deconv_relu(35,16,4,stride=2)
        self.deconv_1 = deconv_sigmoid(19,3,4,stride=2)

        self.predict_8 = torch.nn.Conv2d(64,3,3,stride=1,padding=1)
        self.predict_7 = torch.nn.Conv2d(67,3,3,stride=1,padding=1)
        self.predict_6 = torch.nn.Conv2d(67,3,3,stride=1,padding=1)
        self.predict_5 = torch.nn.Conv2d(67,3,3,stride=1,padding=1)
        self.predict_4 = torch.nn.Conv2d(67,3,3,stride=1,padding=1)
        self.predict_3 = torch.nn.Conv2d(67,3,3,stride=1,padding=1)
        self.predict_2 = torch.nn.Conv2d(35,3,3,stride=1,padding=1)

        self.up_sample_8 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,(3,4),stride=(1,2),padding=1,bias=False),
            torch.nn.Sigmoid()
        )

        self.up_sample_7 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )

        self.up_sample_6 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )

        self.up_sample_5 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )

        self.up_sample_4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )


    def encoder(self, x):
        conv1_out = self.conv_stack1(x)
        conv2_out = self.conv_stack2(conv1_out)
        conv3_out = self.conv_stack3(conv2_out)
        conv4_out = self.conv_stack4(conv3_out)
        conv5_out = self.conv_stack5(conv4_out)
        conv6_out = self.conv_stack6(conv5_out)
        conv7_out = self.conv_stack7(conv6_out)
        conv8_out = self.conv_stack8(conv7_out)


        return conv8_out

    def decoder(self, x):

        deconv8_out = self.deconv_8(x)
        predict_8_out = self.up_sample_8(self.predict_8(x))

        concat_7 = torch.cat([deconv8_out, predict_8_out], dim=1)
        deconv7_out = self.deconv_7(concat_7)
        predict_7_out = self.up_sample_7(self.predict_7(concat_7))

        concat_6 = torch.cat([deconv7_out,predict_7_out],dim=1)
        deconv6_out = self.deconv_6(concat_6)
        predict_6_out = self.up_sample_6(self.predict_6(concat_6))

        concat_5 = torch.cat([deconv6_out,predict_6_out],dim=1)
        deconv5_out = self.deconv_5(concat_5)
        predict_5_out = self.up_sample_5(self.predict_5(concat_5))

        concat_4 = torch.cat([deconv5_out,predict_5_out],dim=1)
        deconv4_out = self.deconv_4(concat_4)
        predict_4_out = self.up_sample_4(self.predict_4(concat_4))

        concat_3 = torch.cat([deconv4_out,predict_4_out],dim=1)
        deconv3_out = self.deconv_3(concat_3)
        predict_3_out = self.up_sample_3(self.predict_3(concat_3))

        concat2 = torch.cat([deconv3_out,predict_3_out],dim=1)
        deconv2_out = self.deconv_2(concat2)
        predict_2_out = self.up_sample_2(self.predict_2(concat2))

        concat1 = torch.cat([deconv2_out,predict_2_out],dim=1)
        predict_out = self.deconv_1(concat1)
        return predict_out
        

    def forward(self,x, reconstructed_latent, refine_latent):
        if refine_latent != True:
            latent = self.encoder(x)
            out = self.decoder(latent)
            return out, latent
        else:
            latent = self.encoder(x)
            out = self.decoder(reconstructed_latent)
            return out, latent        
class AE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=6):
        super(AE, self).__init__()
        self.CNN_AE = EncoderDecoder64x1x1(in_channels)
        self.encoder_output = nn.Linear(64, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 64)
        
    def encoder(self, x):
        cnn_out = self.CNN_AE.encoder(x).flatten(start_dim=1)
        y=self.encoder_output(cnn_out)

        return y
    
    def decoder(self, x):
        x= self.decoder_input(x)
        x = self.CNN_AE.decoder(x.unsqueeze(-1).unsqueeze(-1))
        return x
    
    def forward(self,x):
        return x
import dynlearner as ln  


###############
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
    return regu.mean()*2



################
def Iden(x):
    return x

class CAE_hyp(ln.nn.DynNN):
    '''Adding Constraints to Latent Dynamics of the Autoencoder.
    '''
    def __init__(self, in_channels=3, latent_dim=4, lam=1, latent_dyn='vp',
                 h=0.1, layers=2, width=128, activation='tanh', initializer='orthogonal', 
                 integrator='explicit midpoint', steps=1,
                 
                ):
        super(CAE_hyp, self).__init__()
        self.latent_dim= latent_dim
        self.ae=EncoderDecoder_largekernal(in_channels)
        if latent_dim==64:
            self.encoder_output = Iden
            self.decoder_input= Iden
        else:
            self.encoder_output = nn.Linear(64, latent_dim)
            self.decoder_input = nn.Linear(latent_dim, 64)

        self.lam = lam
        self.latent_dyn=latent_dyn 
        
        
        if self.latent_dyn=='vp':
            vp_LAlayers = 3
            vp_LAsublayers = 2 
            vp_activation = 'sigmoid'
            self.vpnet = ln.nn.LAVPNet(latent_dim, vp_LAlayers, vp_LAsublayers, vp_activation)
 
    
    def regularization(self, x=None, y=None):
        con_loss = []
        for conv in self.ae.regu_modules:
            # print( conv.modules()('0'))
            for module in conv.modules():
                if type(module) is torch.nn.Conv2d and module.weight.shape[-1]>10:
                    con_loss.append(continuous_regu(module.weight))
                
        return sum(con_loss)
                
                


    def latent_forward(self, x0):
        if self.latent_dyn=='vp': 
            x=self.vpnet(x0) 
        return x            

    def encoder(self, x):
        cnn_out = self.ae.encoder(x).flatten(start_dim=1)
        y=self.encoder_output(cnn_out)
        return y
    
    def decoder(self, x):
        x= self.decoder_input(x)
        x = self.ae.decoder(x.unsqueeze(-1).unsqueeze(-1))
        return x       
                   

    def criterion(self, X, y):
        X_latent, y_latent = self.encoder(X), self.encoder(y)           
        y_latent_pre = self.latent_forward(X_latent)
        latent_loss = 2*torch.nn.MSELoss()(y_latent_pre, y_latent) + torch.nn.MSELoss()(self.decoder(y_latent_pre), y)
 
        recon_loss = torch.nn.MSELoss()(self.decoder(X_latent), X) 
        return recon_loss + self.lam * latent_loss       

    def test_criterion(self, X, y):
        X_latent, y_latent = self.encoder(X), self.encoder(y)
        y_latent_pre = self.latent_forward(X_latent)
        latent_loss = torch.nn.MSELoss()(y_latent_pre, y_latent)

        recon_loss = torch.nn.MSELoss()(self.decoder(X_latent), X) #+ torch.nn.MSELoss()(self.decoder(y_latent_pre), y)
        pre_loss = torch.nn.MSELoss()(self.decoder(y_latent_pre), y)

        return recon_loss + self.lam * (latent_loss +  pre_loss)

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
    
  
    
    
class AE_con(ln.nn.DynNN):
    '''Adding Constraints to Latent Dynamics of the Autoencoder.
    '''
    def __init__(self, in_channels=3, latent_dim=4, lam=1, lam_decay=1000, latent_dyn='ode', Inte=False,
                 h=0.1, layers=2, width=128, activation='tanh', initializer='orthogonal', 
                 integrator='explicit midpoint', steps=1,
                 
                ):
        super(AE_con, self).__init__()
        self.latent_dim= latent_dim
        self.ae=EncoderDecoder64x1x1(in_channels)
        self.encoder_output = nn.Linear(64, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 64)

        self.lam = lam
        self.lam_decay=lam_decay
        self.latent_dyn=latent_dyn

        self.Inte = Inte 
        
        if self.latent_dyn=='ode':
            self.odenet = ln.nn.ODENet(dim=latent_dim, h=0.01, layers=3, width=128, activation='tanh', initializer='orthogonal', integrator='explicit midpoint', steps=1)        
        
        if self.latent_dyn=='vp':
            vp_LAlayers = 3
            vp_LAsublayers = 2 
            vp_activation = 'sigmoid'
            self.vpnet = ln.nn.LAVPNet(latent_dim, vp_LAlayers, vp_LAsublayers, vp_activation)
        
  
        
        if self.latent_dyn=='LAsymp':
            symp_LAlayers = 3
            symp_LAsublayers = 2 
            symp_activation = 'sigmoid'
            self.LAsympnet = ln.nn.LASympNet(latent_dim, symp_LAlayers, symp_LAsublayers, symp_activation)  
            
        if self.latent_dyn=='Gsymp':
            symp_Glayers = 2
            symp_Gwidth = 128
            symp_activation = 'sigmoid' 
            self.Gsympnet = ln.nn.GSympNet(latent_dim, symp_Glayers, symp_Gwidth, symp_activation)
        
        if self.latent_dyn=='hnn':
            self.odenet = ln.nn.HNN(dim=latent_dim, h=0.01, layers=3, width=128, activation='tanh', initializer='orthogonal', integrator='explicit midpoint')   
            

    def init_auxi_modu(self):
        if self.latent_dyn=='hnn':
            self.odenet.device = self.device
            self.odenet.dtype = self.dtype
            return 1
        else:
            return 0            
            
        
    def latent_forward(self, x0):
        if self.latent_dyn=='ode': 
             x=self.odenet.ODE_forward(x0)
            # x=self.solver.solve(x0, self.h) 
        if self.latent_dyn=='hnn': 
             x=self.odenet.ODE_forward(x0)            
        if self.latent_dyn=='vp': 
            x=self.vpnet(x0) 
            
        if self.latent_dyn=='LAsymp': 
            x=self.LAsympnet(x0)
            
        if self.latent_dyn=='Gsymp': 
            x=self.Gsympnet(x0)
            
        return x     
          

        
    def encoder(self, x):
        cnn_out = self.ae.encoder(x).flatten(start_dim=1)
        y=self.encoder_output(cnn_out)
        return y
    
    def decoder(self, x):
        x= self.decoder_input(x)
        x = self.ae.decoder(x.unsqueeze(-1).unsqueeze(-1))
        return x                 

    def criterion(self, X, y):
        X_latent, y_latent = self.encoder(X), self.encoder(y)           
        y_latent_pre = self.latent_forward(X_latent)
        latent_loss = torch.nn.MSELoss()(y_latent_pre, y_latent)
        recon_loss = torch.nn.MSELoss()(self.decoder(X_latent), X) + torch.nn.MSELoss()(self.decoder(y_latent), y)
        return self.lam * recon_loss + latent_loss       

    def test_criterion(self, X, y):
        X_latent, y_latent = self.encoder(X), self.encoder(y)
        y_latent_pre = self.latent_forward(X_latent)
        latent_loss = torch.nn.MSELoss()(y_latent_pre, y_latent)

        recon_loss = torch.nn.MSELoss()(self.decoder(X_latent), X) + torch.nn.MSELoss()(self.decoder(y_latent), y)
        pre_loss = torch.nn.MSELoss()(self.decoder(y_latent_pre), y)
        
        print('latent_loss,  recon_loss,  pre_loss:', latent_loss,  recon_loss, pre_loss)
        return self.lam * recon_loss + latent_loss

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
    
    
    ####### ode latent ##########
    def vf(self, x):
        return self.modus['f'](x)
        
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        modules['f'] = ln.nn.FNN(self.dim, self.dim, self.layers, self.width, self.activation, self.initializer)
        return modules 
    
    def integrator_loss(self, x0, x1, h):
        x=self.solver.solve(x0, h)
        return torch.nn.MSELoss()(x1, x)
    def ReturnNp(self, tensor):
        return tensor.cpu().detach().numpy()
    