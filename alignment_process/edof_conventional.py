# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 17:39:20 2022

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


import numpy as np
import mahotas as mh
import cv2
import os
from os.path import isfile, join, isdir
from os import listdir
from utils_align import crop_microscope, alignImageChannels, alignImages

onlyfiles = [f for f in listdir('x3_y15') if isfile(join('x3_y15', f))]


image=[]
for i in range(8):
    img = cv2.imread(join('x3_y15',onlyfiles[i]), cv2.IMREAD_GRAYSCALE)
    mask = np.zeros(img.shape[:2], dtype="uint8")
    #cv2.circle(image, center_coordinates, radius, color, thickness)
    xc=int(img.shape[0]/2)
    yc=int(img.shape[0]/2)
    radius=int(img.shape[0]/2)-15
    cv2.circle(mask, (xc, yc),radius , 255, -1)
    masked = cv2.bitwise_and(img, img, mask=mask)
    
    image.append(masked)


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


print(Tenenbaum(image[2]))

print(Tenenbaum(image[7]))