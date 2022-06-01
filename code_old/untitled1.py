# -*- coding: utf-8 -*-
"""
Created on Mon May 16 10:01:43 2022

@author: albu
"""

from torch.utils.data import Dataset
from torchvision import models, transforms
import pickle
import torch
import numpy as np
import cv2
import glob
from PIL import Image
import os

path = r'C:\Users\albu\Desktop\EDFo_CNN\test_data_aligned'
ext = ['png', 'jpg']    # Add image formats here
files = []
[files.extend(glob.glob(path + '*.' + e)) for e in ext]
X = [np.asarray(Image.open(file)) for file in files]


X_STACKS_per_folder=[]
for idxx,image_name in enumerate(glob.glob(os.path.join(path, "*.jpg"))):
    im = cv2.imread(image_name)
    imnew=cv2.resize(im,(512,512))
    X_STACKS_per_folder.append(imnew)

X = np.array(X_STACKS_per_folder)
X = np.expand_dims(X, axis=0)