# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 19:50:37 2022

@author: albu
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import glob

#calculate Tenenbaum gradient

def Tenenbaum(img):
    
    #Calculate Sobel operators
    sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=3)
    sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=3)
    
    #Calculate Tenenbaum operators with Sobel operators
    tot=sobelx**2+sobely**2
    TENG=sum(sum(tot))
    
    return TENG



direct=r"C:\Users\albu\Documents\GitHub\edof-cnn\test_0"


for idx,image_name in enumerate(glob.glob(r"C:\Users\albu\Documents\GitHub\edof-cnn\test_0\*.png")):
    
    img = cv.imread(image_name,0)
    print(Tenenbaum(img))



image_name='PRED_0.png'
img = cv.imread(image_name,0)
print(Tenenbaum(img))

image_name='GT_0.png'
img = cv.imread(image_name,0)
print(Tenenbaum(img))