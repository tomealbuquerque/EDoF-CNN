# -*- coding: utf-8 -*-
"""
Created on Sat May 28 23:48:12 2022

@author: albu
"""

import numpy as np

# "EDOF_CNN_3D"
for m in (["EDOF_CNN_max","EDOF_CNN_3D","EDOF_CNN_modified"]):

    for s in ([3, 5, 7, 9]):
        mse=[]
        rmse=[]
        ssim=[]
        pnsr=[]
        for i in range(3):
            text_file = open("results\\dataset-cervix93-image_size-640-method-"+str(m)+"-Z-"+str(s)+"-fold-"+str(i)+"-epochs-75-batchsize-2-lr-0.001-cudan-1.txt", "r")
            lines = text_file.readlines()
            mse.append(float(lines[3].split(":")[1]))
            rmse.append(float(lines[4].split(":")[1]))
            ssim.append(float(lines[5].split(":")[1]))
            pnsr.append(float(lines[6].split(":")[1]))
            
        msem=np.mean(mse)
        rmsem=np.mean(rmse)
        ssimm=np.mean(ssim)
        pnsrm=np.mean(pnsr)
        msed=np.std(mse)
        rmsed=np.std(rmse)
        ssimd=np.std(ssim)
        pnsrd=np.std(pnsr)
        
        
        print("&& "+str(s)+" & $"+str(round(msem,5))+" \pm "+ str(round(msed, 5))+"$ & $"+
              str(round(rmsem,5))+" \pm "+ str(round(rmsed, 5))+"$ & $"+
              str(round(ssimm*100,3))+" \pm "+ str(round(ssimd*100, 3))+"$ & $"+
              str(round(pnsrm,3))+" \pm "+ str(round(pnsrd, 3))+"$")