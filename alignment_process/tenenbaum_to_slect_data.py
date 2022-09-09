# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 19:50:37 2022

@author: albu
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import glob
from skimage.metrics import structural_similarity as ssim
from os import listdir
import cv2
import os
from os.path import isfile, join, isdir
import pandas as pd
import numpy as np 
import pickle
from sklearn.model_selection import StratifiedKFold
import random
import shutil

#calculate Tenenbaum gradient

def Tenenbaum(img):
    
    #Calculate Sobel operators
    sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=3)
    sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=3)
    
    #Calculate Tenenbaum operators with Sobel operators
    tot=sobelx**2+sobely**2
    TENG=sum(sum(tot))
    
    return TENG



# 

n_stacks=5

mypath=r"E:\DATA_SETS\new_data_stacks\data_5_stacks_fh - FULL"

geral_dir = [f for f in listdir(mypath) if isdir(join(mypath, f))]

axis_path=[]
img_path=[]
images_list=[]
labels_list=[]
dataframe=[]
i=0
best_all=[]
# geral_dir=[f"E:\DATA_SETS\data_{n_stacks}_stacks_fh\Focus_C313_19-pt1"]
for i in range(len(geral_dir)):
    axis_path=join(mypath,geral_dir[i]) 
    axis_dir = [f for f in listdir(axis_path) if isdir(join(axis_path, f))]
    #create new folders
    best_image=0
    best_sum=[]
    for ad in axis_dir:
        print("axis:",ad)
        path=join(mypath,geral_dir[i], ad)
        best_image=0
        tmp_metric=[]
        for idx,image_name in enumerate(glob.glob(join(path,"*.jpg"))):
            img = cv.imread(image_name,0)
            print(Tenenbaum(img))
            tmp_metric.append(Tenenbaum(img))
        if tmp_metric!=[]:
            if max(tmp_metric[:-1])<tmp_metric[-1]:
                best_image+=1
        if best_image==0:
            shutil.rmtree(path)
            
        best_sum.append(best_image)
        
    if best_sum!=[]:
        print(sum(best_sum),"/",len(axis_dir))
        best_all.append(sum(best_sum))

print(sum(best_all))
# image_name=r'E:\DATA_SETS\data_7_stacks_fh\Focus_C313_19-pt1\x15_y30\output.jpg'
# img = cv.imread(image_name,0)
# print(Tenenbaum(img))


# image_name=r'E:\DATA_SETS\data_7_stacks_fh\Focus_C313_19-pt1\x15_y30\5044.jpg'
# img1 = cv.imread(image_name,0)
# print(Tenenbaum(img1))

# print(ssim(img, img1))

# for idx,image_name in enumerate(glob.glob(r"E:\DATA_SETS\data_5_stacks_fh\Focus_C313_19-pt1\x3_y30\*.jpg")):
#     img = cv.imread(image_name,0)
#     print(Tenenbaum(img))