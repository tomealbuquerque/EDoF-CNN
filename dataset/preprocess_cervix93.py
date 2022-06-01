"""
Preprocessing code for EDoF-CNN using Cervix 93 database

X-> X_STACKS
Y-> Y_EDF (for each group of X_stacks)
"""


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--Z', choices=[3,5,7,9], type=int, default=9)
parser.add_argument('--folds', type=int, choices=range(10),default=3)
parser.add_argument('--img_size', type=int, choices=[224,512,640,960],default=640)
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


#Save only x stack images per EDF
#num images (ni) -> Focused-ni to focused+ni+1  ||| 2->5  3->7 4->9
if args.Z==3:
    ni=1
elif args.Z==5:
    ni=2
elif args.Z==7:
    ni=3
elif args.Z==9:
    ni=4

Y_EDF = []
for idx,image_name in enumerate(glob.glob(r"data\EDF\*.png")):
    im = cv2.imread(image_name)
    imnew=cv2.resize(im,(args.img_size,args.img_size))
    Y_EDF.append(imnew)

Y_EDF = np.array(Y_EDF)

X_STACKS = []
for idx,folder_name in enumerate(glob.glob(r"data\*")):
    if (folder_name!='data\EDF') and (folder_name!='data\labels.csv'):
        X_STACKS_per_folder=[]
        for idxx,image_name in enumerate(glob.glob(os.path.join(folder_name, "*.png"))):
            im = cv2.imread(image_name)
            imnew=cv2.resize(im,(args.img_size,args.img_size))
            X_STACKS_per_folder.append(imnew)
            
        num_images_per_EDF=len(X_STACKS_per_folder)
        limit=round(num_images_per_EDF/2)
        X_STACKS_per_folder=X_STACKS_per_folder[limit-ni:limit+ni+1]       
        X_STACKS.append(X_STACKS_per_folder)
    
X_STACKS = np.array(X_STACKS)

X = np.array(X_STACKS)
Y = np.array(Y_EDF)

# # kfold
state = np.random.RandomState(1234)
kfold = KFold(args.folds, shuffle=True, random_state=state)
folds = [{'train': (X[tr], Y[tr]), 'test': (X[ts], Y[ts])} 
    for tr, ts in kfold.split(X, Y)]
pickle.dump(folds, open(f'data_cervix93_zstacks_'+str(args.Z)+'.pickle', 'wb'))
