# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 15:57:21 2022

@author: albu
"""

import numpy, imageio, elasticdeform
X = numpy.zeros((200, 300))
X[::10, ::10] = 1

# apply deformation with a random 3 x 3 grid
X_deformed = elasticdeform.deform_random_grid(X, sigma=25, points=3)

imageio.imsave('test_X.png', X)
imageio.imsave('test_X_deformed.png', X_deformed)

