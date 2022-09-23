# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 11:02:50 2022

@author: albu
"""

import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from os import listdir
import cv2
import os
from os.path import isfile, join, isdir
import pandas as pd
import numpy as np 
import pickle
from sklearn.model_selection import StratifiedKFold

from utils_align import crop_microscope, alignImageChannels, alignImages

# mypath=r"E:\DATA_SETS\data_5_stacks_fh"
mypath=r"E:\DATA_SETS\data_fraunhofer_no_rgb_align_elastic_aligned_edof"
geral_dir = [f for f in listdir(mypath) if isdir(join(mypath, f))]

axis_path=[]
img_path=[]
images_list=[]
labels_list=[]
dataframe=[]
i=0

for i in range(len(geral_dir)):
    axis_path=join(mypath,geral_dir[i]) 
    axis_dir = [f for f in listdir(axis_path) if isdir(join(axis_path, f))]
    for ad in axis_dir:
        path=join(mypath,geral_dir[i], ad)
    for z in range(len(axis_dir)):
        img_path=join(axis_path,axis_dir[z])
        onlyfiles = [f for f in listdir(img_path) if isfile(join(img_path, f))]
        images=[join(img_path,l) for l in onlyfiles]
        images_list=np.concatenate((images_list, images), axis=0)
        if images!=[]:
            for imna in images:
                im1 = cv2.imread(imna)
                mask = np.zeros(im1.shape[:2], dtype="uint8")
                #cv2.circle(image, center_coordinates, radius, color, thickness)
                xc=int(im1.shape[0]/2)
                yc=int(im1.shape[0]/2)
                radius=int(im1.shape[0]/2)-25
                cv2.circle(mask, (xc, yc),radius , 255, -1)
                masked = cv2.bitwise_and(im1, im1, mask=mask)
                filename = imna.split(os.sep)[-1]
                cv2.imwrite(imna, masked)
                
                
# im1 = cv2.imread(r"E:\DATA_SETS\data_7_stacks_fh\Focus_C313_19-pt1\x3_y30\output.jpg")
# mask = np.zeros(im1.shape[:2], dtype="uint8")
# #cv2.circle(image, center_coordinates, radius, color, thickness)
# xc=int(im1.shape[0]/2)
# yc=int(im1.shape[0]/2)
# radius=int(im1.shape[0]/2)-20
# cv2.circle(mask, (xc, yc),radius , 255, -1)
# masked = cv2.bitwise_and(im1, im1, mask=mask)
# cv2.imwrite(r"E:\DATA_SETS\data_7_stacks_fh\Focus_C313_19-pt1\x3_y30\output.jpg", masked)
