# -*- coding: utf-8 -*-
"""
Compare RGB histograms after clip
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image, ImageOps
import PIL

im = cv2.imread(r'C:\Users\albu\Documents\GitHub\edof-cnn\dataset\data_fraunhofer\Focus_C313_19-pt1\x15_y24\5030.jpg')
# calculate mean value from RGB channels and flatten to 1D array
vals = im.mean(axis=2).flatten()
# plot histogram with 255 bins
b, bins, patches = plt.hist(vals, 255)
plt.xlim([0,255])
plt.ylim([0,20000])
plt.show()



def automatic_brightness_and_contrast(x,clip_hist_perc):
        clip_hist_perc = clip_hist_perc

        gray = cv2.cvtColor(np.float32(x), cv2.COLOR_BGR2GRAY)
        
        # Calculate grayscale histogram
        hist = cv2.calcHist([gray],[0],None,[256],[0,256])
        hist_size = len(hist)
        
        # Calculate cumulative distribution from the histogram
        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index -1] + float(hist[index]))
        
        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_perc *= (maximum/100.0)
        
        clip_hist_perc /= 2.0

        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] <  clip_hist_perc:
            minimum_gray += 1
        
        # Locate right cut
        maximum_gray = hist_size -1
        while accumulator[maximum_gray] >= (maximum -  clip_hist_perc):
            maximum_gray -= 1
        
        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha
        auto_result = cv2.convertScaleAbs(np.float32(x), alpha=alpha, beta=beta)
        
        image=Image.fromarray(np.uint8(auto_result)*255)
        R, G, B = image.split()
        new_image = PIL.Image.merge("RGB", (R, G, B))
        new_image = ImageOps.invert(new_image)
        return new_image

new=automatic_brightness_and_contrast(im,50)

open_cv_image = np.array(new) 
# Convert RGB to BGR 
open_cv_image = open_cv_image[:, :, ::-1].copy()

vals = open_cv_image.mean(axis=2).flatten()
# plot histogram with 255 bins
b, bins, patches = plt.hist(vals, 255)
plt.xlim([0,255])
plt.ylim([0,20000])
plt.show()