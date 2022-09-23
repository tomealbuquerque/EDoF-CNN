# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 18:05:18 2022

@author: albu
"""
#https://github.com/almarklein/pyelastix

import pyelastix
# To read the image data we use imageio
import imageio
# Pick one lib to visualize the result, matplotlib or visvis
#import visvis as plt
import matplotlib.pyplot as plt
import os
import cv2

im_full=[]

im_stacks=['4990.jpg','4992.jpg','4996.jpg','4998.jpg']
# im1_name='4996.jpg'
im2_name='4994.jpg'
for im1_name in im_stacks:
    for i in range(3):
        # im1 = cv2.imread('4994.jpg', cv2.cv2.IMREAD_GRAYSCALE)
        # im2 = cv2.imread('4992.jpg', cv2.cv2.IMREAD_GRAYSCALE)
        # Read image data
        im1 = imageio.imread(im1_name)
        im2 = imageio.imread(im2_name)
        #im2 = imageio.imread('https://dl.dropboxusercontent.com/u/1463853/images/chelsea_morph1.png')
        
        # Select one channel (grayscale), and make float
        im1 = im1[:,:,i].astype('float32')
        im2 = im2[:,:,i].astype('float32')
        
        # im1 = im1[:,:,0].astype('float32')
        # im2 = im2[:,:,0].astype('float32')
        # Get default params and adjust
        params = pyelastix.get_default_params()
        params.NumberOfResolutions = 3
        params.MaximumNumberOfIterations = 500
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
        # cv2.imwrite(filename, im3)
        im_full.append(im3)
    # import pyelastix
    merged = cv2.merge([im_full[2], im_full[1], im_full[0]])
    cv2.imwrite(os.path.basename(im1_name)+'_merged.jpg', merged)
    im_full=[]

# import numpy as np
# import pylab as plt

# # Create some sample data
# dx = np.linspace(0,531,20)
# X,Y = np.meshgrid(dx,dx)
# Z  = field[0]**2 - field[1]
# Z2 = field[0]

# # plt.imshow(Z)
# # plt.colorbar()

# plt.quiver(field[0],field[1],Z2,width=.01,linewidth=1)
# plt.colorbar() 

# plt.show()



# yd, xd = np.gradient(field[0])

# import matplotlib.pyplot as plt
# import numpy as np
# import math
# function_to_plot = lambda x, y: x**2 + y**2
# horizontal_min, horizontal_max, horizontal_stepsize = -2, 3, 0.3
# vertical_min, vertical_max, vertical_stepsize = -1, 4, 0.5

# horizontal_dist = horizontal_max-horizontal_min
# vertical_dist = vertical_max-vertical_min

# horizontal_stepsize = horizontal_dist / float(math.ceil(horizontal_dist/float(horizontal_stepsize)))
# vertical_stepsize = vertical_dist / float(math.ceil(vertical_dist/float(vertical_stepsize)))

# xv, yv = np.meshgrid(np.arange(horizontal_min, horizontal_max, horizontal_stepsize),
#                      np.arange(vertical_min, vertical_max, vertical_stepsize))
# xv+=horizontal_stepsize/2.0
# yv+=vertical_stepsize/2.0

# result_matrix = function_to_plot(xv, yv)
# yd, xd = np.gradient(result_matrix)

# def func_to_vectorize(x, y, dx, dy, scaling=0.01):
#     plt.arrow(x, y, dx*scaling, dy*scaling, fc="k", ec="k", head_width=0.06, head_length=0.1)

# vectorized_arrow_drawing = np.vectorize(func_to_vectorize)

# plt.imshow(np.flip(result_matrix,0), extent=[horizontal_min, horizontal_max, vertical_min, vertical_max])
# vectorized_arrow_drawing(xv, yv, xd, yd, 0.1)
# plt.colorbar()
# plt.show()