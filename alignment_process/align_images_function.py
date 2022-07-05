# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 18:15:00 2022

@author: albu
"""

from __future__ import print_function
import cv2
import numpy as np
import glob
import os

MAX_FEATURES = 2000
GOOD_MATCH_PERCENT = 0.5

def alignImages(im1, im2):

  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)

  # Sort matches by score
  matches= list(matches)
  matches.sort(key=lambda x: x.distance, reverse=False)

  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]

  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  cv2.imwrite("matches.jpg", imMatches)

  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)

  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

  # Use homography
  height, width, channels = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))

  return im1Reg, h



for idx,image_name in enumerate(glob.glob("corrected\*.jpeg")):
    basename = os.path.basename(image_name).split('.')[0]
    
    imReference = cv2.imread(r'corrected\5044_corrected.jpeg')
    im = cv2.imread(image_name)
    imReg, h = alignImages(im, imReference)
    outFilename = basename+"_aligned.jpeg"
    print("Saving aligned image : ", outFilename);
    cv2.imwrite(os.path.join("aligned", outFilename), imReg)