# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 18:12:23 2022

@author: albu
"""

import numpy as np

def Tenenbaum(im):
        
    # compute sharpness metric based on the Tenenbaum gradient
           
    # define the 2D sobel gradient operator
    sobel = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])
    
    sobel3D(:, :, 1) = [[1 2 1],
                        [2 4 2], 
                        [1 2 1]]
    sobel3D(:, :, 2) = zeros(3);
    sobel3D(:, :, 3) = -sobel3D(:, :, 1);
                
            % compute metric
            s = convn(im, sobel3D).^2 + convn(im, permute(sobel3D, [3 1 2])).^2 +  convn(im, permute(sobel3D, [2 3 1])).^2;
            s = sum(s(:));
        
    return s    
    
from matplotlib.image import imread
image = imread(r'C:\Users\albu\Documents\GitHub\edof-cnn\dataset\data\frame000_stack\fov000.png')
     
s = Tenenbaum(image)
    # case 3
            
        #     % define the 3D sobel gradient operator
        #     sobel3D(:, :, 1) = [1 2 1; 2 4 2; 1 2 1];
        #     sobel3D(:, :, 2) = zeros(3);
        #     sobel3D(:, :, 3) = -sobel3D(:, :, 1);
            
        #     % compute metric
        #     s = convn(im, sobel3D).^2 + convn(im, permute(sobel3D, [3 1 2])).^2 +  convn(im, permute(sobel3D, [2 3 1])).^2;
        #     s = sum(s(:));
            