import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
plt.close()

imgL = cv2.imread('im0.png',1)
imgR = cv2.imread('im1.png',1)

'''
DECIMATE
'''
def decimateRGBpic(image, decimatingFactor):
	start = time.time()
	rows = np.shape(image)[0]
	cols = np.shape(image)[1]
	newRows = np.arange(0, rows, decimatingFactor)
	newCols = np.arange(0, cols, decimatingFactor)
	img_smol = image[newRows,:,:]
	img_smol = img_smol[:,newCols,:]
	end = time.time()
	deltaT = end - start
	return img_smol, deltaT

print("\n Decimating Left and Right Eye")
decimation = 7
imgL_smol, timeL = decimateRGBpic(imgL,decimation)
imgR_smol, timeR = decimateRGBpic(imgR,decimation)
print("Decimation Time: "+str(timeL+timeR)+' s')

#show images in their new native size
# cv2.imshow('dL',imgL_smol)
# cv2.imshow('dR',imgR_smol)

'''
DETECT CORNERS
'''
def detectCornersWithShiAndTomasi(gryImage, N, thresh, minDist):
	start = time.time()
	corners = cv2.goodFeaturesToTrack(imgL_smol_gry,N,thresh,minDist)
	corners = np.int0(corners)
	corners = corners[:,0,:]
	#used for viewing where the points are
	# for i in range(0, np.shape(corners)[0]):
	#     cv2.circle(imgL_smol_gry,(corners[i,0],corners[i,1]),3,0,-1)
	end = time.time()
	shiTime = end - start
	return corners, shiTime

#find corners
print("\n Finding Corners")
imgL_smol_gry = cv2.cvtColor(imgL_smol, cv2.COLOR_BGR2GRAY)
corners, shiTime = detectCornersWithShiAndTomasi(imgL_smol_gry,N=500,thresh=0.01,minDist=20)
print("Corner algorithm time: "+str(shiTime)+ ' s')

'''
MAKE REFERENCE WINDOW
'''
#making a reference window from a point found in the corner detector
#soon this will iterate through all points found
thisCorner = 4 #which corner value we are using
windowSideSize = 20
rowCenter = corners[thisCorner,1]
print(rowCenter)
colCenter = corners[thisCorner,0]
print(colCenter)
h = int(windowSideSize/2)
print(h)
refWindow = imgL_smol_gry[rowCenter-h:rowCenter+h,colCenter-h:colCenter+h]
print((rowCenter-h,rowCenter+h,colCenter-h,colCenter+h))
print(np.shape(refWindow))

'''
RUN REFERENCE WINDOW OVER OTHER IMAGE WITH SLIDING WINDOW FUNCTION
'''
def basicSlidingWindow(gryInputImage_matrix, window, differenceMeasure, debugRow=0, debugCol=0):
	'''
	window must be square
	'''
	start = time.time()
	side = np.shape(window)[0]
	half = int(np.ceil(side/2))
	if (np.shape(window)[0] == np.shape(window)[1]):
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
			matchingPoint = np.array([np.unravel_index(np.argmin(resultImage), resultImage.shape)[0]+half,np.unravel_index(np.argmin(resultImage), resultImage.shape)[1]+half])
	else:
		print("ERROR: point was near edge of picture")
		resultImage = np.zeros((np.shape(gryInputImage_matrix)[0]-side,np.shape(gryInputImage_matrix)[1]-side))
		matchingPoint = np.array([0,0])
	end = time.time()
	slideTime = end - start
	return resultImage, matchingPoint, slideTime

print("\n Run sliding window using Mean Squared Error")
imgR_smol_gry = cv2.cvtColor(imgR_smol, cv2.COLOR_BGR2GRAY)
mseResult, foundPoint, slideTime = basicSlidingWindow(imgR_smol_gry, refWindow, 'mse')
mseResult = 255-mseResult
print("sliding window time for one point: "+str(slideTime)+' s')

plt.figure(1)
plt.imshow(mseResult, cmap='gray')
print("                       row  col")
print("wanted point        :",(rowCenter,colCenter))
print("Matching point found:",(foundPoint[0],foundPoint[1]))

'''
make a figure with 2 plots with the left image's selected
points and the right image's found points in corresponding
colors
'''
#convert rgb pictures to "rgb"
imgL_smol_gry_rgb = cv2.cvtColor(imgL_smol_gry, cv2.COLOR_GRAY2RGB)
imgR_smol_gry_rgb = cv2.cvtColor(imgR_smol_gry, cv2.COLOR_GRAY2RGB)

plt.figure(2)
cv2.circle(imgL_smol_gry_rgb,(colCenter,rowCenter),3,(255,255,0),2)
cv2.circle(imgR_smol_gry_rgb,(foundPoint[1],foundPoint[0]),3,(255,255,0),2)
plt.subplot(1,2,1)
plt.imshow(imgL_smol_gry_rgb, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(imgR_smol_gry_rgb, cmap='gray')