# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 00:03:18 2022

@author: albu
"""

import cv2

image1 = cv2.imread(r"C:\Users\albu\Documents\GitHub\edof-cnn\GT_0.png")
image2 = cv2.imread(r"C:\Users\albu\Documents\GitHub\edof-cnn\PRED_0.png")
image3 = cv2.imread(r"C:\Users\albu\Documents\GitHub\edof-cnn\teste_0_stack_2.png") 
image4 = cv2.subtract(image3, image1)

cv2.imshow('image',100-image4)
cv2.waitKey(0)