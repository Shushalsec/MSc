# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 16:59:54 2018

@author: Shushan
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import stats

#	Parse command-line arguments
#	sets K, inputName & outputName
#if len(sys.argv) < 4:
#	print ("Error: Insufficient arguments, imageSegmentation takes three arguments")
#	sys.exit()
#else:
#	K = int(sys.argv[1])
#	inputName = sys.argv[2]
#	outputName = sys.argv[3]



#get the coordinates of a cell pixel as a reference point for later clustering and choosing the correct class 
kernel = np.ones((5,5),np.uint8)
img = cv2.imread('1.jpg')

img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

ix,iy = -1,-1
# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),5,(0,0,255),-1)
        ix,iy = x,y

cv2.namedWindow('Choose a cell')
cv2.setMouseCallback('Choose a cell',draw_circle)
coordinates = list()
while(1):
    cv2.imshow('Choose a cell',img)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('a'):
        coordinates.append((ix,iy))
cv2.destroyAllWindows()




def kmeans(img_file_name, K):
    img = cv2.imread(img_file_name)
    Z = img.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
#    center = np.uint8(center)
#    res = center[label.flatten()]
    #get the image size of the original image
    img_dims = img.shape[0:2] 
    #create an image with the same dimentions as the original image but with pixels with 
    # true value only when they are classified in the claster corresponding to cells
    segmented = label.reshape(img_dims) 
#    plt.gray()
#    plt.imshow(segmented==K-1)
    cell_cluster_options = np.array([segmented[c] for c in coordinates])
    print(cell_cluster_options)
    cell_cluster = stats.mode(cell_cluster_options)[0][0]
    return segmented, cell_cluster

def find_nodes(k_means_segmented, K, cell_cluster, grid_size, threshold):
    '''
    put a grid and compute the number of pixels with true value in each cell
    grid size is the x and y dimension of a cell in the grid
    threshold is the minimum number of black pixels needed to consider the grid element as a node
    '''
    lcopy = np.copy(k_means_segmented) 
    dims = lcopy.shape
    #loop over the cells in the grid
    for i in range(0, dims[0], grid_size):
        for j in range(0, dims[1], grid_size): 
            #compute the number of 'cell' pixels
            s = np.sum(k_means_segmented[i:i+grid_size, j:j+grid_size]==cell_cluster)
            fraction_foreground = s/grid_size*grid_size
            #compare to the teshold value
            if fraction_foreground>threshold:
                lcopy[i-grid_size:i,j-grid_size:j] = 1
            else:
                lcopy[i-grid_size:i,j-grid_size:j] = 0
    
#    plt.imshow(lcopy[:i, :j])
    return (lcopy[:i, :j])






K = 4
segmented, cell_cluster = kmeans('1.jpg', K)
nodes = find_nodes(segmented, K, cell_cluster, 4, 0.7)
plt.subplot(221)
plt.imshow(img)
plt.axis('off')
plt.title('Original image')
plt.subplot(222)
plt.imshow(segmented)
plt.axis('off')
plt.title('K={} segmented image'.format(K))
plt.gray()
plt.subplot(223)
plt.imshow(segmented==cell_cluster)
plt.title('Image with class {} pixels only'.format(cell_cluster))
plt.subplot(224)
plt.imshow(nodes)
plt.axis('off')
plt.title('thresholded image')

