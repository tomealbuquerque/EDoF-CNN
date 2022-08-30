# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 18:05:18 2022

@author: albu
"""
#https://github.com/almarklein/pyelastix

import pyelastix
import imageio
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import pandas as pd
import pickle
import _pickle as cPickle
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
import glob
from os import listdir
from os.path import isfile, join, isdir


mypath=f"E:\DATA_SETS\low_cost_microscopy_dataset\data_fraunhofer"
mypath_new=f"E:\DATA_SETS\low_cost_microscopy_dataset\data_fraunhofer_elastic_aligned"
os.makedirs(mypath_new, exist_ok=True)

geral_dir = [f for f in listdir(mypath) if isdir(join(mypath, f))]

axis_path=[]

for gd in range(len(geral_dir)):
    axis_path=join(mypath,geral_dir[gd]) 
    axis_dir = [f for f in listdir(axis_path) if isdir(join(axis_path, f))]
    for ad in axis_dir:
        path=join(mypath,geral_dir[gd], ad)
        # for idx,image_name in enumerate(glob.glob(join(path,"*.jpg"))):
        image_names=[image_name for idx,image_name in enumerate(glob.glob(join(path,"*.jpg")))]
        
        im_central_name=image_names[2]
        im_stacks=[image_names[0],image_names[1],image_names[3],image_names[4]]
        # img = cv2.imread(image_name)

        #create folders for new images
        path_new_dir = os.path.join(mypath_new,geral_dir[gd], ad)
        os.makedirs(path_new_dir, exist_ok=True)
        
        im_full=[]
        im_ref_full=[]
        for im1_name in im_stacks:
            for i in range(3): #for every rgb channel
                # Read image data
                im1 = imageio.imread(im1_name)
                im2 = imageio.imread(im_central_name)
        
                # Select one channel (grayscale), and make float
                im1 = im1[:,:,i].astype('float32')
                im2 = im2[:,:,i].astype('float32')
        
                # Get default params and adjust
                params = pyelastix.get_default_params()
                params.NumberOfResolutions = 2
                params.MaximumNumberOfIterations = 100
                # print(params)
                
                # Register!
                im3, field = pyelastix.register(im1, im2, params,True)
                
                im_full.append(im3)
                im_ref_full.append(im2)
                
            # save elastic aligned images
            merged = cv2.merge([im_full[2], im_full[1], im_full[0]])
            cv2.imwrite(os.path.join(mypath_new,geral_dir[gd], ad,os.path.basename(im1_name)), merged)
            merged_ref = cv2.merge([im_ref_full[2], im_ref_full[1], im_ref_full[0]])
            cv2.imwrite(os.path.join(mypath_new,geral_dir[gd], ad,os.path.basename(im_central_name)), merged_ref)
            im_full=[]
            im_ref_full=[]
