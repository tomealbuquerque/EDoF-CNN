# -*- coding: utf-8 -*-
"""
Created on Mon May 30 20:42:26 2022

@author: albu
"""


import numpy as np
import cv2
import os 
import glob



from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, normalized_root_mse 


Phat=[]
Y_true=[]
for i in range(31):
    for idx,path in enumerate(glob.glob(r"test_images_for_EDOF\image_"+str(i)+"\*")):
        # print(path)
        file_name = os.path.basename(path)
        if os.path.basename(path).split('_')[0]== 'Output':
            # print(file_name)
            im = cv2.imread(path)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)/255
            Phat.append(im)
            
for i in range(31):
    path = r"test_images_for_EDOF\\full_edof\\image_full_edof_"+str(i)+".png"
    # print(path)
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)/255
    Y_true.append(im)  
            
            


mse = np.mean([mean_squared_error(Y_true[i], Phat[i]) for i in range(len(Y_true))])
rmse = np.mean([normalized_root_mse(Y_true[i], Phat[i]) for i in range(len(Y_true))])
ssim =np.mean([ssim(Y_true[i], Phat[i],channel_axis=0) for i in range(len(Y_true))]) 
psnr =np.mean([peak_signal_noise_ratio(Y_true[i], Phat[i]) for i in range(len(Y_true))]) 



# f = open('results\\'+ str(prefix)+'.txt', 'a+')
# f.write('\n\nModel:'+str(prefix)+
#     ' \nMSE:'+ str(mse)+
#     ' \nRMSE:'+ str(rmse)+
#     ' \nSSIM:'+str(ssim)+
#     ' \nPSNR:'+ str(psnr))
# f.close()