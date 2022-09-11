"""
Preprocessing code for EDoF-CNN using Fraunhofer database

X-> X_STACKS
Y-> Y_EDF (for each group of X_stacks)
"""


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--Z', choices=[5], type=int, default=5)
parser.add_argument('--folds', type=int, choices=range(5),default=5)
parser.add_argument('--img_size', type=int, choices=[224,512],default=512)
args = parser.parse_args()


import numpy as np
import pandas as pd
import glob
import cv2
import pickle
import os
import _pickle as cPickle
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
import glob
from os import listdir
from os.path import isfile, join, isdir


mypath=r"E:\DATA_SETS\data_fraunhofer_no_rgb_align_elastic_aligned_edof"

geral_dir = [f for f in listdir(mypath) if isdir(join(mypath, f))]

axis_path=[]
Y_EDF=[]
X_STACKS=[]
X_stacks_full=[]

for i in range(len(geral_dir)):
    axis_path=join(mypath,geral_dir[i]) 
    axis_dir = [f for f in listdir(axis_path) if isdir(join(axis_path, f))]
    for ad in axis_dir:
        path=join(mypath,geral_dir[i], ad)
        for idx,image_name in enumerate(glob.glob(join(path,"*.jpg"))):
            img = cv2.imread(image_name)
            img=cv2.resize(img,(args.img_size,args.img_size))
            if os.path.basename(image_name)=='Output.jpg':
                Y_EDF.append(img)

            else:
                X_STACKS.append(img)
                
            
        if os.path.exists(join(path,"output.jpg")):
            pass 
        else:
                print(path)
        if join(path,"output.jpg")==[]:
            print(path)
            
        X_stacks_full.append(X_STACKS)
        X_STACKS=[]

Y_EDF = np.array(Y_EDF)
X_STACKS = np.array(X_stacks_full)

X = np.array(X_STACKS)
Y = np.array(Y_EDF)


# # kfold
state = np.random.RandomState(1234)
kfold = KFold(args.folds, shuffle=True, random_state=state)
folds = [{'train': (X[tr], Y[tr]), 'test': (X[ts], Y[ts])} 
    for tr, ts in kfold.split(X, Y)]
pickle.dump(folds, open(f'data_fraunhofer_no_rgb_elastic.pickle', 'wb'))
