# -*- coding: utf-8 -*-
"""
Created on Sat May 28 16:54:59 2022

@author: albu
"""
from PIL import Image, ImageOps
import PIL

new_image=Image.open("teste0.png")
new_image = ImageOps.invert(new_image)

new_image.show()