# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 15:29:03 2022

@author: albu
"""
from os import listdir
import cv2
import os
from os.path import isfile, join, isdir
import pandas as pd
import numpy as np 
import pickle
from sklearn.model_selection import StratifiedKFold

from utils_align import crop_microscope, alignImageChannels, alignImages

mypath=r"E:\DATA_SETS\cyto_quality_fh"
mynewpath=r"E:\DATA_SETS\data_channel_aligned"

geral_dir = [f for f in listdir(mypath) if isdir(join(mypath, f))]

axis_path=[]
img_path=[]
images_list=[]
labels_list=[]
dataframe=[]
i=0

for i in range(len(geral_dir)):
    axis_path=join(mypath,geral_dir[i]) 
    axis_path_new=join(mynewpath,geral_dir[i]) 
    axis_dir = [f for f in listdir(axis_path) if isdir(join(axis_path, f))]
    #create new folders
    for ad in axis_dir:
        path=join(mynewpath,geral_dir[i], ad)
        path_old=join(mypath,geral_dir[i], ad)
        if not os.listdir(path_old):
            print("empty")
            os.rmdir(path_old)
        else:
            pass
        
