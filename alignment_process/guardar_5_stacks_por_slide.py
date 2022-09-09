# -*- coding: utf-8 -*-
"""
Create the new dataset with 7 stacks per slide
"""
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

# random.seed(12)
nimages_slide_to_save = 20
stacks_to_save=5

mypath=r"E:\DATA_SETS\new_data_stacks\data_channel_position_aligned - Copy"
mynewpath=r"E:\DATA_SETS\low_cost_microscopy_dataset\data_channel_aligned_position_static_aligned"

os.makedirs(mynewpath, exist_ok=True)

geral_dir = [f for f in listdir(mypath) if isdir(join(mypath, f))]

axis_path=[]
img_path=[]
images_list=[]
labels_list=[]
dataframe=[]

for i in range(len(geral_dir)):
    axis_path=join(mypath,geral_dir[i]) 
    axis_path_new=join(mynewpath,geral_dir[i]) 
    axis_dir = [f for f in listdir(axis_path) if isdir(join(axis_path, f))]
    #turn on for random selection
    # axis_dir=random.choices(axis_dir, k=nimages_slide_to_save)
    #create new folders
    for ad in axis_dir:
        path=join(mynewpath,geral_dir[i], ad)
        os.makedirs(path, exist_ok=True)
    for z in range(len(axis_dir)):
        img_path=join(axis_path,axis_dir[z])
        onlyfiles = [f for f in listdir(img_path) if isfile(join(img_path, f))]
        images=[join(img_path,l) for l in onlyfiles if l!='focused_image.txt']
        images_list=np.concatenate((images_list, images), axis=0)
        if images!=[]:
            onlyfiles[-1]=='focused_image.txt'
            focus_img=open(join(img_path,onlyfiles[-1])).read()
            label_path=focus_img+'.jpg'
            path_focus_img=join(img_path,label_path)
            focus_number=[i for i in range(len(images)) if images[i] == path_focus_img][0]  

            if stacks_to_save==5:
                stkl=2
                stkr=3
            else:
                stkl=3
                stkr=4
                
            images=images[focus_number-stkl:focus_number+stkr]
            
            for imna in images:
                im = cv2.imread(imna)
                filename = imna.split(os.sep)[-1]
                outFilename=join(axis_path_new,axis_dir[z],filename)
                cv2.imwrite(outFilename, im)
                
        print('remaining:',len(axis_dir)-z-1,'/',len(axis_dir))
        