# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 18:05:18 2022

@author: albu
"""


import pyelastix
# To read the image data we use imageio
import imageio
# Pick one lib to visualize the result, matplotlib or visvis
#import visvis as plt
import matplotlib.pyplot as plt

import cv2

im_full=[]

for i in range(3):
    # im1 = cv2.imread('4994.jpg', cv2.cv2.IMREAD_GRAYSCALE)
    # im2 = cv2.imread('4992.jpg', cv2.cv2.IMREAD_GRAYSCALE)
    # Read image data
    im1 = imageio.imread('4996.jpg')
    im2 = imageio.imread('4992.jpg')
    #im2 = imageio.imread('https://dl.dropboxusercontent.com/u/1463853/images/chelsea_morph1.png')
    
    # Select one channel (grayscale), and make float
    im1 = im1[:,:,i].astype('float32')
    im2 = im2[:,:,i].astype('float32')
    
    # im1 = im1[:,:,0].astype('float32')
    # im2 = im2[:,:,0].astype('float32')
    # Get default params and adjust
    params = pyelastix.get_default_params()
    params.NumberOfResolutions = 3
    print(params)
    
    # Register!
    im3, field = pyelastix.register(im1, im2, params,True)
    
    # Visualize the result
    fig = plt.figure(1);
    plt.clf()
    plt.subplot(231); plt.imshow(im1)
    plt.subplot(232); plt.imshow(im2)
    plt.subplot(234); plt.imshow(im3)
    plt.subplot(235); plt.imshow(field[0])
    plt.subplot(236); plt.imshow(field[1])
    
    # Enter mainloop
    if hasattr(plt, 'use'):
        plt.use().Run()  # visvis
    else:
        plt.show()  # mpl
    
    
    # Filename
    filename = 'savedImage'+str(i)+'.png'
      
    # Using cv2.imwrite() method
    # Saving the image
    cv2.imwrite(filename, im3)
    im_full.append(im3)
# import pyelastix
merged = cv2.merge([im_full[2], im_full[1], im_full[0]])
cv2.imwrite('merged.jpg', merged)

cv2.imshow("Merged", merged)
cv2.waitKey(0)
cv2.destroyAllWindows()

