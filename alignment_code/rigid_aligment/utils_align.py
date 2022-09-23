from __future__ import print_function
import cv2
import numpy as np
import glob
import os


MAX_FEATURES = 2000
GOOD_MATCH_PERCENT = 0.9

def crop_microscope(img_to_crop):
    pad_y = img_to_crop.shape[0]//200 
    pad_x = img_to_crop.shape[1]//200
    img = img_to_crop[pad_y:-pad_y, pad_y:-pad_y,:]
 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,50,255,cv2.THRESH_BINARY) 
    x,y,w,h = cv2.boundingRect(thresh) #getting crop points
    
    
#since we cropped borders we need to uncrop it back 
    if y!=0: 
        y = y+pad_y
    if h == thresh.shape[0]:
        h = h+pad_y
    if x !=0:
        x = x +pad_x
    if w == thresh.shape[1]:
        w = w + pad_x
    h = h+pad_y
    w = w + pad_x
    crop = img_to_crop[y:y+h,x:x+w]
    
    return crop



def alignImageChannels(im1):

  im1=crop_microscope(im1)

  red_channel = im1[:,:,2]
  blue_channel = im1[:,:,1]
  green_channel = im1[:,:,0]
  

  # Convert images to grayscale
  im1Gray = red_channel
  im2Gray = blue_channel
  im3Gray = green_channel
  # Detect ORB features and compute descriptors.
  
  orb = cv2.ORB_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
  keypoints3, descriptors3 = orb.detectAndCompute(im3Gray, None)

  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  
  matches = matcher.match(descriptors1, descriptors2, None)
  
  matches2 = matcher.match(descriptors3, descriptors2, None)

  # Sort matches by score
  matches= list(matches)
  matches.sort(key=lambda x: x.distance, reverse=False)
  
  matches2= list(matches2)
  matches2.sort(key=lambda x: x.distance, reverse=False)
  
  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]
  
  numGoodMatches2 = int(len(matches2) * GOOD_MATCH_PERCENT)
  matches2 = matches2[:numGoodMatches2]

  # Draw top matches
  imMatches = cv2.drawMatches(red_channel, keypoints1, blue_channel, keypoints2, matches, None)
  cv2.imwrite("matches.jpg", imMatches)
  
  imMatches2 = cv2.drawMatches(green_channel, keypoints3, blue_channel, keypoints2, matches2, None)
  cv2.imwrite("matches2.jpg", imMatches2)

  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)
  
  points22 = np.zeros((len(matches2), 2), dtype=np.float32)
  points3 = np.zeros((len(matches2), 2), dtype=np.float32)

  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
    
  for i, match in enumerate(matches2):
    points3[i, :] = keypoints3[match.queryIdx].pt
    points22[i, :] = keypoints2[match.trainIdx].pt


  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

  # Use homography
  height, width = red_channel.shape
  im1Reg = cv2.warpPerspective(red_channel, h, (width, height))
  
  
  h, mask = cv2.findHomography(points3, points22, cv2.RANSAC)
  
  height, width = green_channel.shape
  im2Reg = cv2.warpPerspective(green_channel, h, (width, height))
  
  

  im1[:,:,2] = im1Reg
  im1[:,:,1] = blue_channel
  im1[:,:,0] = im2Reg

  mask = np.zeros(im1.shape[:2], dtype="uint8")
  #cv2.circle(image, center_coordinates, radius, color, thickness)
  xc=int(im1.shape[0]/2)
  yc=int(im1.shape[0]/2)
  radius=int(im1.shape[0]/2)-15
  cv2.circle(mask, (xc, yc),radius , 255, -1)
  masked = cv2.bitwise_and(im1, im1, mask=mask)
  
  masked=crop_microscope(masked)


  return masked, h


GOOD_MATCH_PERCENT_align = 0.9

MAX_FEATURES_align = 10000

def alignImages(im1, im2):
  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_FEATURES_align)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)

  # Sort matches by score
  matches= list(matches)
  matches.sort(key=lambda x: x.distance, reverse=False)

  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT_align)
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
