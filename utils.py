# =============================================================================
# Utils files for the models 
# Inspired on: https://github.com/tyui592/Perceptual_loss_for_real_time_style_transfer
# =============================================================================

import torch
import torch.nn as nn


class ConvLayer(nn.Module):    
    def __init__(self, in_ch, out_ch, kernel_size, stride, pad='reflect', activation='relu', normalization='instance'):        
        super(ConvLayer, self).__init__()
        
        # padding
        if pad == 'reflect':            
            self.pad = nn.ReflectionPad2d(kernel_size//2)
        elif pad == 'zero':
            self.pad = nn.ZeroPad2d(kernel_size//2)
        else:
            raise NotImplementedError("Not expected pad flag !!!")
    
            
        # convolution
        self.conv_layer = nn.Conv2d(in_ch, out_ch, 
                                    kernel_size=kernel_size,
                                    stride=stride)
        
        # activation
        if activation == 'relu':
            self.activation = nn.ReLU()        
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'linear':
            self.activation = lambda x : x
        else:
            raise NotImplementedError("Not expected activation flag !!!")

        # normalization 
        if normalization == 'instance':            
            self.normalization = nn.InstanceNorm2d(out_ch, affine=True)
        else:
            raise NotImplementedError("Not expected normalization flag !!!")

    def forward(self, x):
        x = self.pad(x)
        x = self.conv_layer(x)
        x = self.normalization(x)
        x = self.activation(x)        
        return x
    
    
class ResidualLayer(nn.Module):    
    def __init__(self, in_ch, out_ch, kernel_size, stride, pad='zero', normalization='instance'):        
        super(ResidualLayer, self).__init__()
        
        self.conv1 = ConvLayer(in_ch, out_ch, kernel_size, stride, pad, 
                               activation='relu', 
                               normalization=normalization)
        
        self.conv2 = ConvLayer(out_ch, out_ch, kernel_size, stride, pad, 
                               activation='linear', 
                               normalization=normalization)
        
    def forward(self, x):
        y = self.conv1(x)
        return self.conv2(y) + x
        
    
class DeconvLayer(nn.Module):    
    def __init__(self, in_ch, out_ch, kernel_size, stride, pad='reflect', activation='relu', normalization='instance', upsample='bicubic', align_corners=True):        
        super(DeconvLayer, self).__init__()
        
        # upsample
        self.upsample = upsample
        
        # pad
        if pad == 'reflect':            
            self.pad = nn.ReflectionPad2d(kernel_size//2)
        elif pad == 'zero':
            self.pad = nn.ZeroPad2d(kernel_size//2)
        else:
            raise NotImplementedError("Not expected pad flag !!!")        
        
        # conv
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride)
        
        # activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'linear':
            self.activation = lambda x : x
        else:
            raise NotImplementedError("Not expected activation flag !!!")
        
        # normalization
        if normalization == 'instance':
            self.normalization = nn.InstanceNorm2d(out_ch, affine=True)
        else:
            raise NotImplementedError("Not expected normalization flag !!!")
        
    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode=self.upsample)        
        x = self.pad(x)
        x = self.conv(x)
        x = self.normalization(x)        
        x = self.activation(x)
        return x


