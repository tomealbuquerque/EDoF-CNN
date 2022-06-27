# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 19:50:37 2022

@author: albu
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import glob

direct=r"C:\Users\albu\Documents\GitHub\edof-cnn\dataset\data\frame000_stack"


for idx,image_name in enumerate(glob.glob(r"C:\Users\albu\Documents\GitHub\edof-cnn\dataset\data\frame000_stack\*.png")):
    
    img = cv.imread(image_name,0)
    laplacian = cv.Laplacian(img,cv.CV_64F)
    sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=3)
    sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=3)
    
    tot=sobelx**2+sobely**2
    Tenenbaum=sum(sum(tot))
    
    print(Tenenbaum)
