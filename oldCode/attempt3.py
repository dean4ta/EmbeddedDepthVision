# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 19:37:54 2018

@author: dean4ta
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
plt.close()

imgL = cv2.imread('Data/Cable-perfect/Cable-perfect/im0.png',1)
imgR = cv2.imread('Data/Cable-perfect/Cable-perfect/im1.png',1)
gry_imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
gry_imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
RGB_imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
RGB_imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)

cv2.imwrite('Original.png',imgL)
cv2.imwrite('Original_R.png',imgR)

#text = input("press enter to show original plots: \n")


#plot original
#cv2.imshow('something',imgL)
# plt.imshow(RGB_imgL)
# plt.show()

######################
######################
'''Corner detectors'''
######################

########## Harris corners ##########
img = cv2.imread('Data/Cable-perfect/Cable-perfect/im0.png',1)
print("\n Harris Corner")

start = time.time()
gray = np.float32(gry_imgL)
dst = cv2.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,255,255]
end = time.time()

# plt.imshow(img)
# plt.show()
harrisTime = end - start
print(harrisTime)
cv2.imwrite('Original_harrisCorner.png',img)


####################################
########## Shi and Tomasi ##########
img = cv2.imread('Data/Cable-perfect/Cable-perfect/im0.png',1)
img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#bilateral filtered image
img_bi = cv2.imread('Data/Cable-perfect/Cable-perfect/im0.png',1)

start = time.time()
img_bi = cv2.bilateralFilter(img_bi,9,50,50)
end = time.time()
bilateral = end - start

img_bi_gry = cv2.cvtColor(img_bi, cv2.COLOR_BGR2GRAY)

N = 500
minDist = 25
thresh = 0.01

print("\n Shi and Tomasi")
print("bilateral filter time: " + str(bilateral))

#shi and tomasi on original image
start = time.time()
corners = cv2.goodFeaturesToTrack(img_gry,N,thresh,minDist)
corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),7,255,-1)
end = time.time()
shiTime = end - start
print("original: " + str(shiTime))

#shi and tomasi on bilaterally filtered image
start = time.time()
corners2 = cv2.goodFeaturesToTrack(img_bi_gry,N,thresh,minDist)
corners2 = np.int0(corners2)
for i in corners2:
    x,y = i.ravel()
    cv2.circle(img_bi,(x,y),7,255,-1)
end = time.time()
shiTime2 = end - start


print("bilateral: " + str(shiTime2))

plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(img_bi)
plt.show()
cv2.imwrite('Original_ShiAndTomasi.png',img)
cv2.imwrite('Original_bilateral_ShiAndTomasi.png',img_bi)

###########################################
########## Level curve curvature ##########

#######################################################
########## Hessian feature strength measures ##########

###########################
########## SUSAN ##########

##########################
########## FAST ##########
print("\n FAST")

img = cv2.imread('Data/Cable-perfect/Cable-perfect/im0.png',0)
img_rgb = cv2.imread('Data/Cable-perfect/Cable-perfect/im0.png',1)

start = time.time()
# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()
# find and draw the keypoints
kp = fast.detect(img,None)
img2 = cv2.drawKeypoints(img_rgb, kp, None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
end = time.time()
fastTime = end - start
print("fast: " + str(shiTime))

# Print all default params
# print("Threshold: ", fast.getInt('threshold'))
# print("nonmaxSuppression: ", fast.getBool('nonmaxSuppression'))
# print("neighborhood: ", fast.getInt('type'))
# print("Total Keypoints with nonmaxSuppression: ", len(kp))


#fast with bilateral
img = cv2.imread('Data/Cable-perfect/Cable-perfect/im0.png',0)
img_rgb = cv2.imread('Data/Cable-perfect/Cable-perfect/im0.png',1)

img = cv2.bilateralFilter(img,8,25,25)

start = time.time()
# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()
# find and draw the keypoints
kp_bi = fast.detect(img,None)
img2_bi = cv2.drawKeypoints(img_rgb, kp_bi, None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
end = time.time()
fastTime = end - start
print("fast: " + str(fastTime))

# Print all default params
# print("Threshold: ", fast.getInt('threshold'))
# print("nonmaxSuppression: ", fast.getBool('nonmaxSuppression'))
# print("neighborhood: ", fast.getInt('type'))
# print("Total Keypoints with nonmaxSuppression: ", len(kp))

cv2.imwrite('Original_fast.png',img2)
cv2.imwrite('Original_bilateral_fast.png',img2_bi)

plt.figure(2)
plt.subplot(1,2,1)
plt.imshow(img2)
plt.subplot(1,2,2)
plt.imshow(img2_bi)
plt.show()

'''
# Disable nonmaxSuppression
fast.setBool('nonmaxSuppression',0)
kp = fast.detect(img,None)

print("Total Keypoints without nonmaxSuppression: ", len(kp))

img3 = cv2.drawKeypoints(img, kp, color=(255,0,0))

cv2.imwrite('fast_false.png',img3)
'''
##########################
########## SIFT ##########
print('\n SIFT')

img = cv2.imread('Data/Cable-perfect/Cable-perfect/im0.png')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
start = time.time()
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)

img=cv2.drawKeypoints(gray,kp,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
end = time.time()
siftTime = end - start
print("sift: " + str(siftTime))

cv2.imwrite('Original_sift.png',img)
plt.figure(3)
plt.subplot(1,2,1)
plt.imshow(img)

#SIFT with bilateral
img = cv2.imread('Data/Cable-perfect/Cable-perfect/im0.png',0)
img_rgb = cv2.imread('Data/Cable-perfect/Cable-perfect/im0.png',1)

img = cv2.bilateralFilter(img,8,25,25)
start = time.time()
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)

img=cv2.drawKeypoints(gray,kp,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
end = time.time()
siftTime2 = end - start
print("sift bilateral: " + str(siftTime2))

cv2.imwrite('Original_bilateral_sift.png',img)
plt.subplot(1,2,2)
plt.imshow(img)


#https://docs.opencv.org/3.3.0/db/d27/tutorial_py_table_of_contents_feature2d.html
#use link for more feature descriptors


'''Ridge Detection'''
#Hough Transform

#Radon Transform

'''Finding Correspondance Points with snake sliding window'''
'''
psuedocode:
for p in interestingPoints:
	get coordinates from interestingPoints[i]
	for window in windows:

'''


#MSE

#PSNR

#SSIM