# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:55:03 2022

@author: albu
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:49:16 2022

@author: albu
"""

import pickle
import numpy as np
import torch 
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

fold=0
type='test'
X, Y = pickle.load(open(f'data_CERVIX93.pickle', 'rb'))[fold][type]



# from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
# # X: (N,3,H,W) a batch of non-negative RGB images (0~255)
# # Y: (N,3,H,W)  
# Y = np.transpose( Y, (0, 3, 1, 2))
# X = np.transpose( X, (0,1, 4, 2, 3))

# Y1=[]
# for i in range(5):
#     Y1.append(Y[0])
    
    
# X = torch.from_numpy(np.array(X))
# Y = torch.from_numpy(np.array(Y1))

# # calculate ssim & ms-ssim for each image
# ssim_val = ssim( X[4], Y, data_range=255, size_average=False) # return (N,)
# ms_ssim_val = ms_ssim(  X[0], Y, data_range=255, size_average=False ) #(N,)

# values=ssim_val.cpu().numpy()


# # set 'size_average=True' to get a scalar value as loss. see tests/tests_loss.py for more details
# ssim_loss = 1 - ssim( X, Y, data_range=255, size_average=True) # return a scalar
# ms_ssim_loss = 1 - ms_ssim( X, Y, data_range=255, size_average=True )

# # reuse the gaussian kernel with SSIM & MS_SSIM. 
# ssim_module = SSIM(data_range=255, size_average=True, channel=3) # channel=1 for grayscale images
# ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3)

# ssim_loss = 1 - ssim_module(X[1], Y)
# ms_ssim_loss = 1 - ms_ssim_module(X, Y)

# import torch

# from torchvision import transforms
# from PIL import Image


# # transform = transforms.ToTensor()

# # x = transform(im1).unsqueeze(0).cuda() # .cuda() for GPU
# # y = transform(im2).unsqueeze(0).cuda()

# from piqa import psnr, ssim

# print('PSNR:', psnr.psnr(X[0][0], Y[0],))
# # print('SSIM:', ssim.ssim(x, y))

X1 = torch.tensor(np.array(X[0][1]),dtype=float)
Y1 = torch.tensor(np.array(Y[0]),dtype=float)
loss = nn.MSELoss()
output = loss(X1, Y1)
loss_mse=output.cpu().numpy()