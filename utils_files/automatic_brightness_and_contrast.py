import numpy as np
import cv2
from PIL import Image, ImageOps
import PIL


class automatic_brightness_and_contrast:
    """Rotate by one of the given angles."""

    def __init__(self, clip_hist_perc):
        self.clip_hist_perc = clip_hist_perc

    def __call__(self, x):
        # x=np.array(x) 
        self.clip_hist_percent=self.clip_hist_perc 
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
        self.clip_hist_percent *= (maximum/100.0)
        
        self.clip_hist_percent /= 2.0

        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] <  self.clip_hist_percent:
            minimum_gray += 1
        
        # Locate right cut
        maximum_gray = hist_size -1
        while accumulator[maximum_gray] >= (maximum -  self.clip_hist_percent):
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

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)