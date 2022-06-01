import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchgeometry.losses import DiceLoss, ssim
from utils import ConvLayer, ResidualLayer, DeconvLayer

import pytorch_ssim

mse = nn.MSELoss()
mae = nn.L1Loss()
ssim_loss = pytorch_ssim.SSIM()
l1 = nn.L1Loss(reduction='sum')



def total_variation_loss(img, weight):
      bs_img, c_img, h_img, w_img = img.size()
      tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
      tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
      return weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)

#ver total variation loss 
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
        input_max, max_indices= torch.max(torch.stack(Enc),dim=0,keepdim=False)
        RS = self.residual(input_max)
        Dec = self.decoder(RS)
        return Dec
    
    def loss(self, Yhat, Y):
        return 0.1*mse(Yhat, Y) + total_variation_loss(Y,Yhat) + ssim_loss(Yhat, Y)
    



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


class EDOF_CNN_2(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_2, self).__init__()      
        model = models.resnet18(pretrained=False)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
        model = nn.Sequential(*tuple(model.children())[:-1])
        
        # model = getattr(models, pretrained_model)(pretrained=True)
        
        self.encoder = nn.Sequential(    
            model)

        self.residual = nn.Sequential(            
            ResidualLayer(512, 512, 3, 1),
            ResidualLayer(512, 512, 3, 1),
            ResidualLayer(512, 512, 3, 1),
            ResidualLayer(512, 512, 3, 1),
            ResidualLayer(512, 512, 3, 1))
            
        self.decoder = nn.Sequential( 
            DeconvLayer(512, 64, 3, 1),
            DeconvLayer(64, 32, 3, 1),
            DeconvLayer(32, 16, 3, 1),
            DeconvLayer(16, 8, 3, 1),
            DeconvLayer(8, 4, 3, 1, activation='relu'),
            ConvLayer(4, 1, 1, 1, activation='linear'))
        
    def forward(self, XX):
        Enc = [self.encoder(X) for X in XX]
        input_max, max_indices= torch.max(torch.stack(Enc),dim=0,keepdim=False)
        print(input_max.shape)
        RS = self.residual(input_max)
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
            ConvLayer(64, 64, 3, 2))
        
        self.residual = nn.Sequential(            
            ResidualLayer(448, 448, 3, 1),
            ResidualLayer(448, 448, 3, 1),
            ResidualLayer(448, 448, 3, 1),
            ResidualLayer(448, 448, 3, 1),
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
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=2, padding_mode='replicate') # padding mode same as original Caffe code
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=2, padding_mode='replicate')
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2, padding_mode='replicate')
        
    def forward(self, XX):
        Enc = [self.encoder(X) for X in XX]
        input_cat = torch.stack(Enc)
        RS = self.residual(input_cat)
        Dec = self.decoder(RS)
        x = F.relu(self.conv1(Dec))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y) + total_variation_loss(Y,Yhat) + ssim_loss(Yhat, Y)