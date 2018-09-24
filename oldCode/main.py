import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

print("Testing Sobel")
img = cv.imread('Data/Cable-perfect/Cable-perfect/im0.png',0)

laplacian = cv.Laplacian(img,cv.CV_64F)
sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)

#method 1
#sobelx_norm = (sobelx-np.min(sobelx))*255/np.max(sobelx-np.min(sobelx))
#sobelx_norm = sobelx_norm.astype(np.uint8)

#method 2
#sobelx_norm = sobelx-np.min(sobelx)

#method 3
#sobelx_norm = np.abs(sobelx)
sobelx2 = sobelx


plt.subplot(2,3,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(2,3,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

plt.subplot(2,3,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])

plt.subplot(2,3,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.subplot(2,3,5),plt.imshow(sobelx2,cmap = 'gray')
plt.title('Sobel X normalized'), plt.xticks([]), plt.yticks([])



plt.imshow(img,cmap = 'gray')


#%%



#%%