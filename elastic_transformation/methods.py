# # -*- coding: utf-8 -*-
# """
# Created on Mon Aug  8 18:05:18 2022

# @author: albu
# """


# import pyelastix
# import imageio
# import matplotlib.pyplot as plt

# import cv2

# im1 = imageio.imread('4990.jpg')
# im2 = imageio.imread('4994.jpg')


# im_full=[]
# im_full_fields=[]

# for i in range(3):
#     # Read image data

#     # Select one channel (grayscale), and make float
#     im1i = im1[:,:,i].astype('float32')
#     im2i = im2[:,:,i].astype('float32')
    
    
#     # Get default params and adjust
#     params = pyelastix.get_default_params()
#     params.NumberOfResolutions = 3
#     print(params)
    
#     # Register!
#     im3, field = pyelastix.register(im1i, im2i, params,True)
    
#     # Visualize the result
#     # fig = plt.figure(1);
#     # plt.clf()
#     # plt.subplot(231); plt.imshow(im1)
#     # plt.subplot(232); plt.imshow(im2)
#     # plt.subplot(234); plt.imshow(im3)
#     # plt.subplot(235); plt.imshow(field[0])
#     # plt.subplot(236); plt.imshow(field[1])
    
#     # # Enter mainloop
#     # if hasattr(plt, 'use'):
#     #     plt.use().Run()  # visvis
#     # else:
#     #     plt.show()  # mpl
    
#     imgf=field[0]
#     # plt.axis('off')
#     # Filename
#     # filename = 'savedImage'+str(i)+'.png'
      
#     # Using cv2.imwrite() method
#     # Saving the image
#     # cv2.imwrite(filename, im3)
#     im_full.append(im3)
#     im_full_fields.append(imgf)
    
    
# merged = cv2.merge([im_full[2], im_full[1], im_full[0]])
# cv2.imwrite('merged.jpg', merged)

# merged_fields = cv2.merge([im_full_fields[2], im_full_fields[1], im_full_fields[0]])

# cv2.imwrite('merged_fields.jpg', merged_fields)


# cv2.imshow("Merged", merged)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import pyelastix

# To read the image data we use imageio
import imageio

# Pick one lib to visualize the result, matplotlib or visvis
#import visvis as plt
import matplotlib.pyplot as plt


# Read image data
im1 = imageio.imread('images\\4990.jpg')
im2 = imageio.imread('images\\4994.jpg')
#im2 = imageio.imread('https://dl.dropboxusercontent.com/u/1463853/images/chelsea_morph1.png')

# Select one channel (grayscale), and make float
im1 = im1[:,:,1].astype('float32')
im2 = im2[:,:,1].astype('float32')

# Get default params and adjust
params = pyelastix.get_default_params()
params.NumberOfResolutions = 3
print(params)

# Register!
im3, field = pyelastix.register(im1, im2, params)

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