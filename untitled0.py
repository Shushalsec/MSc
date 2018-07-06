# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 16:59:54 2018

@author: Shushan
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import data, feature


img = cv2.imread('1.jpg')
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 5
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
#get the image size of the original image
dims = img.shape[0:2] 
#create an image with the same dimentions as the original image but with pixels with 
# true value only when they are classified in the claster corresponding to cells
segmented = label.reshape(dims) 
plt.gray()
plt.imshow(segmented==K-1)

#put a grid and compute the number of pixels with true value in each cell
grid_size = 50 #x and y dimension of a cell in the grid
lcopy = np.copy(segmented) 
threshold = 10
#loop over the cells in the grid
for i in range(grid_size, dims[0], grid_size):
    for j in range(grid_size, dims[1], grid_size): 
        #compute the numebr of 'cell' pixels
        s = np.sum(segmented[i-grid_size:i,j-grid_size:j]==K-1)
        #compare to the teshold value
        if s>threshold:
            lcopy[i-grid_size:i,j-grid_size:j] = 1
        else:
            lcopy[i-grid_size:i,j-grid_size:j] = 0

plt.imshow(lcopy)
feature.blob_doh(img[:,:,2]/255)
plt.imshow(lcopy)
