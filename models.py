# =============================================================================
# Models architectures
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchgeometry.losses import DiceLoss, ssim
from utils import ConvLayer, ResidualLayer, DeconvLayer
from einops import rearrange
from utils_files import pytorch_ssim
import kornia.losses


mse = nn.MSELoss()
mae = nn.L1Loss()
ssim_loss = pytorch_ssim.SSIM()
l1 = nn.L1Loss(reduction='sum')

from torch import nn
import torch
from piq import TVLoss

lossTV = TVLoss()

def tv_loss(c):
    x = c[:,:,1:,:] - c[:,:,:-1,:]
    y = c[:,:,:,1:] - c[:,:,:,:-1]
    loss = torch.sum(torch.abs(x)) + torch.sum(torch.abs(y))
    return loss

def compute_total_variation_loss(img, weight):      
    tv_h = ((img[:,:,1:,:] - img[:,:,:-1,:]).pow(2)).sum()
    tv_w = ((img[:,:,:,1:] - img[:,:,:,:-1]).pow(2)).sum()    
    return weight * (tv_h + tv_w)


class EDOF_CNN_fast(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_fast, self).__init__()        
        self.encoder = nn.Sequential(    
            ConvLayer(1, 32, 3, 1),
            ConvLayer(32, 64, 3, 2))
        
        self.residual = nn.Sequential(            
            ResidualLayer(64, 64, 3, 1),
            ResidualLayer(64, 64, 3, 1),
            ResidualLayer(64, 64, 3, 1),
            ResidualLayer(64, 64, 3, 1),
            ResidualLayer(64, 64, 3, 1))
            
        self.decoder = nn.Sequential( 
            DeconvLayer(64, 32, 3, 1),
            DeconvLayer(32, 16, 3, 2, activation='relu'),
            ConvLayer(16, 1, 1, 1, activation='linear'))
        
    def forward(self, XX):
        Enc = [self.encoder(X) for X in XX]
        input_max, max_indices= torch.min(torch.stack(Enc),dim=0,keepdim=False)
        RS = self.residual(input_max)
        Dec = self.decoder(RS)

        return Dec
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y) 
 # + 1e-3 * tv_loss(Yhat.clone().detach())

class EDOF_CNN_max(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_max, self).__init__()        
        self.encoder = nn.Sequential(    
            ConvLayer(1, 32, 3, 1),
            ConvLayer(32, 64, 3, 2),
            ConvLayer(64, 128, 3, 2))
        
        self.residual = nn.Sequential(            
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1))
            
        self.decoder = nn.Sequential( 
            DeconvLayer(128, 64, 3, 1),
            DeconvLayer(64, 32, 3, 1),
            DeconvLayer(32, 16, 3, 2, activation='relu'),
            ConvLayer(16, 1, 1, 1, activation='linear'))
        
        
    def forward(self, XX):
        Enc = [self.encoder(X) for X in XX]
        input_max, max_indices= torch.min(torch.stack(Enc),dim=0,keepdim=False)
        RS = self.residual(input_max)
        Dec = self.decoder(RS)
        return Dec
    
    def loss(self, Yhat, Y): 
        return mse(Yhat, Y)
    
    # + ssim_loss(Yhat,Y)


class EDOF_CNN_3D(nn.Module):    
    def __init__(self,Z):    
        super(EDOF_CNN_3D, self).__init__()        
        self.encoder = nn.Sequential(    
            ConvLayer(Z, 32, 9, 1), #ver quantos z stacks temos
            ConvLayer(32, 64, 3, 2),
            ConvLayer(64, 128, 3, 2)
            )
        
        self.residual = nn.Sequential(            
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1))
            
        self.decoder = nn.Sequential( 
            DeconvLayer(128, 64, 3, 1),
            DeconvLayer(64, 32, 3, 1),
            DeconvLayer(32, 16, 3, 2),
            ConvLayer(16, 1, 1, 1, activation='linear'))
        
        
    def forward(self, XX):
        XXX = torch.squeeze(torch.stack([X for X in XX], dim=2), 1)
        Enc = self.encoder(XXX)
        RS = self.residual(Enc)
        Dec = self.decoder(RS)
        return Dec
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)



class EDOF_CNN_concat(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_concat, self).__init__()        
        self.encoder = nn.Sequential(    
            ConvLayer(1, 32, 3, 1),
            ConvLayer(32, 64, 3, 2),
            ConvLayer(64, 448, 3, 2))
        
        self.residual = nn.Sequential(            
            ResidualLayer(448, 448, 3, 1),
            ResidualLayer(448, 448, 3, 1),
            ResidualLayer(448, 448, 3, 1),
            ResidualLayer(448, 448, 3, 1),
            ResidualLayer(448, 448, 3, 1))

            
        self.decoder = nn.Sequential( 
            DeconvLayer(448, 64, 3, 1),
            DeconvLayer(64, 32, 3, 1),
            DeconvLayer(32, 16, 3, 2, activation='relu'),
            ConvLayer(16, 1, 1, 1, activation='linear'))
        
        #Super resolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=2, padding_mode='replicate')
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=2, padding_mode='replicate')
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2, padding_mode='replicate')
        
    def forward(self, XX):
        Enc = [self.encoder(X) for X in XX]
        input_cat = torch.stack(Enc)
        input_cat = rearrange(input_cat, 'd0 d1 d2 d3 d4 -> d0 (d1 d2) d3 d4')
        RS = self.residual(input_cat)
        Dec = self.decoder(RS)
        x = F.relu(self.conv1(Dec))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)
    
class EDOF_CNN_backbone(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_backbone, self).__init__()      
        model = models.mobilenet_v2(pretrained=True)
        model = nn.Sequential(*tuple(model.children())[:-1])
        # last_dimension = torch.flatten(model(torch.randn(1, 3, 640, 640))).shape[0]

        self.encoder = nn.Sequential(    
            model)

        self.residual = nn.Sequential(
            ConvLayer(1280, 254, 1, 2),            
            ResidualLayer(254, 254, 3, 1),
            ResidualLayer(254, 254, 3, 1),
            ResidualLayer(254, 254, 3, 1),
            # ResidualLayer(128, 128, 3, 1),
            # ResidualLayer(128, 128, 3, 1),
            # ResidualLayer(128, 128, 3, 1),
            # ResidualLayer(128, 128, 3, 1),
            # ResidualLayer(128, 128, 3, 1),
            ResidualLayer(254, 254, 3, 1))
            
        self.decoder = nn.Sequential( 
            DeconvLayer(254, 128, 3, 1),
            DeconvLayer(128, 64, 3, 1),
            DeconvLayer(64, 32, 3, 1),
            DeconvLayer(32, 16, 3, 1),
            DeconvLayer(16, 8, 3, 1),
            DeconvLayer(8, 4, 3, 1),
            DeconvLayer(4, 3, 3, 2, activation='relu'),
            ConvLayer(3, 3, 1, 1, activation='linear'))
        
    def forward(self, XX):
        Enc = [self.encoder(X) for X in XX]
        input_max, max_indices= torch.max(torch.stack(Enc),dim=0,keepdim=False)
        RS = self.residual(input_max)
        Dec = self.decoder(RS)
        return Dec
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y) 
    # + 1e-3 * tv_loss(Yhat.clone().detach())


class EDOF_CNN_RGB(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_RGB, self).__init__()        
        self.encoder = nn.Sequential(    
            ConvLayer(3, 32, 3, 1),
            ConvLayer(32, 64, 3, 2))
        
        
        
        self.residual = nn.Sequential(            
            ResidualLayer(64, 64, 3, 1),
            ResidualLayer(64, 64, 3, 1),
            ResidualLayer(64, 64, 3, 1),
            ResidualLayer(64, 64, 3, 1),
            ResidualLayer(64, 64, 3, 1))
            
        self.decoder = nn.Sequential( 
            DeconvLayer(64, 32, 3, 1),
            DeconvLayer(32, 16, 3, 2, activation='relu'),
            ConvLayer(16, 3, 1, 1, activation='linear'))
        
    def forward(self, XX):
        Enc = [self.encoder(X) for X in XX]
        input_max, max_indices= torch.min(torch.stack(Enc),dim=0,keepdim=False)
        RS = self.residual(input_max)
        Dec = self.decoder(RS)

        return Dec
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y) 



class EDOF_CNN_pairwise(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pairwise, self).__init__()        
        self.encoder = nn.Sequential(    
            ConvLayer(1, 32, 3, 1),
            ConvLayer(32, 64, 3, 2))
        
        self.encoder2 = nn.Sequential(    
            ConvLayer(64, 64, 3, 1),
            ConvLayer(64, 64, 3, 1))
        
        self.residual = nn.Sequential(            
            ResidualLayer(64, 64, 3, 1),
            ResidualLayer(64, 64, 3, 1),
            ResidualLayer(64, 64, 3, 1),
            ResidualLayer(64, 64, 3, 1),
            ResidualLayer(64, 64, 3, 1))
            
        self.decoder = nn.Sequential( 
            DeconvLayer(64, 32, 3, 1),
            DeconvLayer(32, 16, 3, 2, activation='relu'),
            ConvLayer(16, 1, 1, 1, activation='linear'))
        
    def forward(self, XX):
        Enc = [self.encoder(X) for X in XX]
        
        input_max0, max_indices= torch.min(torch.stack(Enc[0:1]),dim=0,keepdim=False)
        input_max2, max_indices= torch.min(torch.stack(Enc[1:3]),dim=0,keepdim=False)
        
        XXX=[input_max0, input_max2]
        
        Enc = [self.encoder2(X) for X in XXX]
        input_max, max_indices= torch.min(torch.stack(Enc),dim=0,keepdim=False)
        RS = self.residual(input_max)
        Dec = self.decoder(RS)

        return Dec
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y) 




#print parameters of models
model_edofmax=EDOF_CNN_max()
model_edof3d =EDOF_CNN_3D(5)
model_edoffast=EDOF_CNN_fast()
model_edofpair=EDOF_CNN_pairwise()
# model_edofmod =EDOF_CNN_modified()
print("EDOF_CNN_max: ",sum(p.numel() for p in model_edofmax.encoder.parameters())*6+sum(p.numel() for p in model_edofmax.parameters()))
print("EDOF_CNN_3D: ",sum(p.numel() for p in model_edof3d.parameters()))
print("EDOF_CNN_fast: ",sum(p.numel() for p in model_edoffast.encoder.parameters())*6+sum(p.numel() for p in model_edoffast.parameters()))
print("EDOF_CNN_pairwise: ",sum(p.numel() for p in model_edofpair.encoder.parameters())*6+sum(p.numel() for p in model_edofpair.parameters()))

