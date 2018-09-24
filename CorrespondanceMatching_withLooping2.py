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
imgL_smol, timeL = decimateRGBpic(imgL,5)
imgR_smol, timeR = decimateRGBpic(imgR,5)
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
REFERENCE WINDOW OVER OTHER IMAGE WITH SLIDING WINDOW FUNCTION
'''
def basicSlidingWindow(gryInputImage_matrix, window, differenceMeasure, debugRow=0, debugCol=0):
	'''
	window must be square
	'''
	#start = time.time()
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
			matchingPoint = np.unravel_index(np.argmin(resultImage), resultImage.shape)[0]+half,np.unravel_index(np.argmin(resultImage), resultImage.shape)[1]+half
	else:
		print("ERROR: point was near edge of picture")
		resultImage = np.zeros((np.shape(gryInputImage_matrix)[0]-side,np.shape(gryInputImage_matrix)[1]-side))
		matchingPoint = 0,0
	#end = time.time()
	#slideTime = end - start
	return resultImage, matchingPoint#, slideTime


print("\n Run sliding window using Mean Squared Error")
'''
MAKE REFERENCE WINDOW
'''
#making a reference window from a point found in the corner detector
#soon this will iterate through all points found
start = time.time()
windowSideSize = 25
h = int(windowSideSize/2)
imgR_smol_gry = cv2.cvtColor(imgR_smol, cv2.COLOR_BGR2GRAY)
foundPoint = np.zeros((np.shape(corners)), dtype=np.int64)
for thisCorner in range(0,np.shape(corners)[0]):
	#thisCorner = 20 #which corner value we are using
	print("corner "+str(thisCorner)+' of '+str(np.shape(corners)[0]))
	rowCenter = corners[thisCorner,1]
	colCenter = corners[thisCorner,0]
	refWindow = imgL_smol_gry[rowCenter-h:rowCenter+h,colCenter-h:colCenter+h]

	'''
	RUN REFERENCE WINDOW OVER OTHER IMAGE WITH SLIDING WINDOW FUNCTION
	'''
	mseResult, tempPoint = basicSlidingWindow(imgR_smol_gry, refWindow, 'mse')
	foundPoint[thisCorner,:] = tempPoint
	mseResult = 255-mseResult
#print("sliding window time for one point: "+str(slideTime)+' s')
end = time.time()
totalTime = end - start
print("sliding window time for all points: "+str(totalTime)+' s')
# plt.figure(1)
# plt.imshow(mseResult, cmap='gray')
# print("                       row  col")
# print("wanted point        :",(rowCenter,colCenter))
# print("Matching point found:",(foundPoint[0],foundPoint[1]))

'''
make a figure with 2 plots with the left image's selected
points and the right image's found points in corresponding
colors
'''
#convert rgb pictures to "rgb"
imgL_smol_gry_rgb = cv2.cvtColor(imgL_smol_gry, cv2.COLOR_GRAY2RGB)
imgR_smol_gry_rgb = cv2.cvtColor(imgR_smol_gry, cv2.COLOR_GRAY2RGB)

plt.figure(2)
# colorWheel = np.array([[0,0,255],
# 					   [0,255,0],
# 					   [0,255,255],
# 					   [255,0,0],
# 					   [255,0,255],
# 					   [255,255,0]])
# colorWheel.astype(uint8)
colorWheel = ((0,0,255),
			  (0,255,0),
			  (0,255,255),
			  (255,0,0),
			  (255,0,255),
			  (255,255,0))

for corner in range(0,np.shape(corners)[0]):
	cv2.circle(imgL_smol_gry_rgb,(corners[corner,0],corners[corner,1]),3,colorWheel[corner%len(colorWheel)],2)
	cv2.circle(imgR_smol_gry_rgb,(foundPoint[corner,1],foundPoint[corner,0]),3,colorWheel[corner%len(colorWheel)],2)
plt.subplot(1,2,1)
plt.imshow(imgL_smol_gry_rgb, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(imgR_smol_gry_rgb, cmap='gray')