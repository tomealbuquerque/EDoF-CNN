import torch
import torch.nn as nn

import torchvision.models as models
import torchvision.transforms as transforms
from torchgeometry.losses import DiceLoss, ssim
from utils import ConvLayer, ResidualLayer, DeconvLayer

mse = nn.MSELoss()
# SSIM = ssim()


def dice_loss(pred, target):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    
    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )

class EDOF_CNN(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN, self).__init__()        
        self.encoder = nn.Sequential(    
            ConvLayer(3, 32, 9, 1),
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
            ConvLayer(32, 3, 9, 1, activation='linear'))
        
    def forward(self, XX):
        Enc = [self.encoder(X) for X in XX]
        input_max, max_indices= torch.max(torch.stack(Enc),dim=0,keepdim=False)
        RS = self.residual(input_max)
        Dec = self.decoder(RS)
        return Dec
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)




