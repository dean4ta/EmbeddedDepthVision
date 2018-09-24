import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
plt.close()



imgL = cv2.imread('Data/Cable-perfect/Cable-perfect/im0.png',1)
imgR = cv2.imread('Data/Cable-perfect/Cable-perfect/im1.png',1)
# gry_imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
# gry_imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
# RGB_imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
# RGB_imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)

# cv2.imwrite('Original.png',imgL)
# cv2.imwrite('Original_R.png',imgR)

#downsample the image to like 480x680 instead of 1984x2796
#decimating
def decimateRGBpic(image, decimatingFactor):
	rows = np.shape(image)[0]
	cols = np.shape(image)[1]
	newRows = np.arange(0, rows, decimatingFactor)
	newCols = np.arange(0, cols, decimatingFactor)
	imgL_smol = image[newRows,:,:]
	imgL_smol = imgL_smol[:,newCols,:]
	return imgL_smol

start = time.time()
decimator = 6
rows = np.shape(imgL)[0]
cols = np.shape(imgL)[1]
newRows = np.arange(0, rows, decimator)
newCols = np.arange(0, cols, decimator)
imgL_smol = imgL[newRows,:,:]
imgL_smol = imgL_smol[:,newCols,:]
end = time.time()
decimateTime = end - start

print('\n decimation time: ' + str(decimateTime))

cv2.imshow('df',imgL_smol)

'''Shi and Tomasi on downsampled image'''
imgL_smol_gry = cv2.cvtColor(imgL_smol, cv2.COLOR_BGR2GRAY)
imgL_smol_gry2 = imgL_smol_gry
N = 500
minDist = 20
thresh = 0.01

print("\nShi and Tomasi")

start = time.time()
corners = cv2.goodFeaturesToTrack(imgL_smol_gry,N,thresh,minDist)
corners = np.int0(corners)
corners = corners[:,0,:]
for i in range(0, np.shape(corners)[0]):
    #print(corners[i])
    #imgL_smol[corners[i]] =
    cv2.circle(imgL_smol_gry2,(corners[i,0],corners[i,1]),3,0,-1)
end = time.time()
shiTime = end - start
print(" time elapsed: " + str(shiTime))

plt.figure(1)
plt.imshow(imgL_smol_gry, cmap='gray')


#20x20 reference image
i = 10 #which corner value we are using
windowSideSize = 16
rowCenter = corners[i,1]
colCenter = corners[i,0]
h = int(windowSideSize/2)
imgL_smol_gry = cv2.cvtColor(imgL_smol, cv2.COLOR_BGR2GRAY)
refWindow = imgL_smol_gry[rowCenter-h:rowCenter+h,colCenter-h:colCenter+h]
plt.figure(2)
plt.subplot(1,2,2)
plt.imshow(refWindow, cmap='gray')
def basicSlidingWindow(gryInputImage_matrix, window, differenceMeasure, debugRow=0, debugCol=0):
	'''
	
	window must be square
	'''
	#basic sliding window
	side = np.shape(window)[0]
	half = int(np.ceil(side/2))
	resultImage = np.zeros((np.shape(gryInputImage_matrix)[0]-side,np.shape(gryInputImage_matrix)[1]-side))
	for row in range(half, np.shape(gryInputImage_matrix)[0]-half):
		for col in range(half, np.shape(gryInputImage_matrix)[1]-half):
			#debugging
			# if row == debugRow and col == debugCol:
			# 	count = 37
			sample = gryInputImage_matrix[row-half:row+half,col-half:col+half]
			#Mean Squared Error
			if differenceMeasure == 'mse':
				measure = np.sum((window-sample)**2)
			#elif differenceMeasure == 'other'
			resultImage[row-half,col-half] = measure  
	if differenceMeasure == 'mse':
		matchingPoint = np.unravel_index(np.argmin(resultImage), resultImage.shape)[0]+half,np.unravel_index(np.argmin(resultImage), resultImage.shape)[1]+half
	return resultImage, matchingPoint

start = time.time()
mseResult, foundPoint = basicSlidingWindow(imgL_smol_gry, refWindow, 'mse')
mseResult = 255-mseResult
end = time.time()
windowTime = end - start
print("\n Sliding Window time: " + str(windowTime))


plt.subplot(1,2,1)
plt.imshow(mseResult, cmap='gray')

print("                       row  col")
print("wanted point        :",(rowCenter,colCenter))
print("Matching point found:",(foundPoint[0],foundPoint[1]))


		

			