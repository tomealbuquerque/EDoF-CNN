# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 19:50:37 2022

@author: albu
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import glob
from skimage.metrics import structural_similarity as ssim

#calculate Tenenbaum gradient

def Tenenbaum(img):
    
    #Calculate Sobel operators
    sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=3)
    sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=3)
    
    #Calculate Tenenbaum operators with Sobel operators
    tot=sobelx**2+sobely**2
    TENG=sum(sum(tot))
    
    return TENG



for idx,image_name in enumerate(glob.glob(r"C:\Users\albu\Documents\GitHub\edof-cnn\to_test_quality\*.png")):
    
    img = cv.imread(image_name,0)
    print(image_name)
    print(Tenenbaum(img))



# image_name=r'E:\DATA_SETS\data_7_stacks_fh\Focus_C313_19-pt1\x15_y30\output.jpg'
# img = cv.imread(image_name,0)
# print(Tenenbaum(img))


# image_name=r'E:\DATA_SETS\data_7_stacks_fh\Focus_C313_19-pt1\x15_y30\5044.jpg'
# img1 = cv.imread(image_name,0)
# print(Tenenbaum(img1))

# print(ssim(img, img1))