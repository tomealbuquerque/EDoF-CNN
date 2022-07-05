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
            with open(join(axis_path_new,axis_dir[z],'focused_image.txt'), 'w') as f:
                f.write(focus_img)
            
            for imna in images:
                im = cv2.imread(imna)
                
                imReg, h = alignImageChannels(im)
                filename = imna.split(os.sep)[-1]
                outFilename=join(axis_path_new,axis_dir[z],filename)
                cv2.imwrite(outFilename, imReg)
        print('remaining:',len(axis_dir)-z-1,'/',len(axis_dir))
        
        
mypath=r"E:\DATA_SETS\data_channel_aligned"
mynewpath=r"E:\DATA_SETS\data_channel_position_aligned"
    
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
            with open(join(axis_path_new,axis_dir[z],'focused_image.txt'), 'w') as f:
                f.write(focus_img)
            
            for imna in images:
                im = cv2.imread(imna)
                imReference = cv2.imread(join(axis_path,axis_dir[z],label_path))
                imReg, h = alignImages(im, imReference)
                filename = imna.split(os.sep)[-1]
                outFilename=join(axis_path_new,axis_dir[z],filename)
                
                cv2.imwrite(outFilename, imReg)
        print('remaining:',len(axis_dir)-z-1,'/',len(axis_dir))
        
        
        
    
        

            