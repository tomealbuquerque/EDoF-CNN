# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 17:00:29 2022

@author: albu
"""

import numpy as np
import mahotas as mh
import cv2
import os
from os.path import isfile, join, isdir
from os import listdir

onlyfiles = [f for f in listdir('x3_y15') if isfile(join('x3_y15', f))]

image=[]
for i in range(3):
    img = cv2.imread(join('x3_y15',onlyfiles[i]), cv2.IMREAD_GRAYSCALE)
    image.append(img)

image=np.array(image)

stack,h,w = image.shape

focus = np.array([mh.sobel(t, just_filter=True) for t in image])

best = np.argmax(focus, 0)

r = np.zeros((h,w))-1
for y in range(h):
    for x in range(w):
        r[y,x] = image[best[y,x], y, x]
        
image = image.reshape((stack,-1)) # image is now (stack, nr_pixels)
image = image.transpose() # image is now (nr_pixels, stack)
r = image[np.arange(len(image)), best.ravel()] # Select the right pixel at each location
r = r.reshape((h,w)) # reshape to get final result

from matplotlib import pyplot as plt
plt.imshow(r, cmap='gray', vmin=0, vmax=255)
plt.show()