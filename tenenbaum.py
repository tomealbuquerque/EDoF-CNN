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



# for idx,image_name in enumerate(glob.glob(r"C:\Users\albu\Documents\GitHub\edof-cnn\to_test_quality\*.png")):
    
#     img = cv.imread(image_name,0)
#     print(image_name)
#     print(Tenenbaum(img))



image_name=r'Output.jpg'
img = cv.imread(image_name,0)
print(Tenenbaum(img))


image_name=r'Output_merged.jpg'
img1 = cv.imread(image_name,0)
print(Tenenbaum(img1))

# print(Tenenbaum(img)-Tenenbaum(img1))


# Cumulative probability of blur detection (CPBD) https://ivulab.asu.edu/software/quality/cpbd
import cpbd

import cv2


input_image1 = cv2.imread('Output.jpg')
input_image1 = cv2.cvtColor(input_image1, cv2.COLOR_BGR2GRAY)
input_image2 = cv2.imread('Output_merged.jpg')
input_image2 = cv2.cvtColor(input_image2, cv2.COLOR_BGR2GRAY)
print("non-elastic alignment:")
print(cpbd.compute(input_image1))
print("elastic alignment:")
print(cpbd.compute(input_image2))

# print(ssim(img, img1))