# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 22:52:33 2017

@author: Rebecca
"""

import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
import cv2
from skimage.feature import hog
from funLib import *
#-----------------------------------------
## load dataset and data overview
#-----------------------------------------
cars = glob.glob('./vehicles/*/*.png')
notcars = glob.glob('./non-vehicles/*/*.png')


# Define a function to return some characteristics of the dataset 
def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    example_img = mpimg.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype
    # Return data_dict
    return data_dict
    
data_info = data_look(cars, notcars)

print('Your function returned a count of', 
      data_info["n_cars"], ' cars and', 
      data_info["n_notcars"], ' non-cars')
print('of size: ',data_info["image_shape"], ' and data type:', 
      data_info["data_type"])
# randomly choose car / not-car indices and plot example images   


car_ind = np.random.randint(0, len(cars))
notcar_ind = np.random.randint(0, len(notcars))
    
# Read in car / not-car images
car_image = mpimg.imread(cars[car_ind])
notcar_image = mpimg.imread(notcars[notcar_ind])

# Plot the examples
fig = plt.figure()
plt.subplot(121)
plt.imshow(car_image)
plt.title('Example Car Image')
plt.subplot(122)
plt.imshow(notcar_image)
plt.title('Example Not-car Image')
plt.savefig('output_images/exampleCarNNoncar.jpg', bbox_inches='tight')
#-----------------------------------------
## explore colorspaces
#-----------------------------------------
car_inds = np.random.randint(0, len(cars),2)
notcar_inds = np.random.randint(0, len(notcars),2)
img0 = mpimg.imread( cars[car_inds[0]])
img1 = mpimg.imread( cars[car_inds[1]])
img2 = mpimg.imread( notcars[notcar_inds[0]])
img3 = mpimg.imread( notcars[notcar_inds[1]])

# show in different color spaces
fig, axs = plt.subplots(7,4,figsize=(40,70))
#fig.subplots_adjust(hspace=0.01,wspace=0.01)
colSpaces = {'RGB','HSV','LUV','YUV','YCrCb','LAB'}
i=0
#for col in colSpaces:
axs[i,0].imshow(img0 ) ,axs[i,0].set_title('car1',     fontsize=50)
axs[i,1].imshow(img1 ) ,axs[i,1].set_title('car2',     fontsize=50)
axs[i,2].imshow(img2 ) ,axs[i,2].set_title('non car1', fontsize=50)
axs[i,3].imshow(img3 ) ,axs[i,3].set_title('non car2', fontsize=50)

i=i+1   
strColor=['R','G','B']
for j in range(3):  
    axs[j+i,0].imshow(img0[:,:,j],cmap='gray'),axs[j+i,0].set_title('RGB--'+strColor[j], fontsize=50)
    axs[j+i,1].imshow(img1[:,:,j],cmap='gray'),axs[j+i,1].set_title('RGB--'+strColor[j], fontsize=50)  
    axs[j+i,2].imshow(img2[:,:,j],cmap='gray'),axs[j+i,2].set_title('RGB--'+strColor[j], fontsize=50)  
    axs[j+i,3].imshow(img3[:,:,j],cmap='gray'),axs[j+i,3].set_title('RGB--'+strColor[j], fontsize=50)  
i=i+3 
p0 = cv2.cvtColor(img0, cv2.COLOR_RGB2HSV)  
p1 = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV) 
p2 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV) 
p3 = cv2.cvtColor(img3, cv2.COLOR_RGB2HSV) 
strColor=['H','S','V']
for j in range(3):  
	axs[j+i,0].imshow(p0[:,:,j],cmap='gray'),axs[j+i,0].set_title('HSV--'+strColor[j], fontsize=50) 
	axs[j+i,1].imshow(p1[:,:,j],cmap='gray'),axs[j+i,1].set_title('HSV--'+strColor[j], fontsize=50)   
	axs[j+i,2].imshow(p2[:,:,j],cmap='gray'),axs[j+i,2].set_title('HSV--'+strColor[j], fontsize=50)   
	axs[j+i,3].imshow(p3[:,:,j],cmap='gray'),axs[j+i,3].set_title('HSV--'+strColor[j], fontsize=50)   

plt.savefig('output_images/exploreColorspaces_RGB_HSV.jpg', bbox_inches='tight')

fig, axs = plt.subplots(7,4,figsize=(40,70))
#fig.subplots_adjust(hspace=0.01,wspace=0.01)

i=0
#for col in colSpaces:
axs[i,0].imshow(img0 ) ,axs[i,0].set_title('car1',     fontsize=50)
axs[i,1].imshow(img1 ) ,axs[i,1].set_title('car2',     fontsize=50)
axs[i,2].imshow(img2 ) ,axs[i,2].set_title('non car1', fontsize=50)
axs[i,3].imshow(img3 ) ,axs[i,3].set_title('non car2', fontsize=50)

i=i+1   
p0 = cv2.cvtColor(img0, cv2.COLOR_RGB2LAB)  
p1 = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB) 
p2 = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB) 
p3 = cv2.cvtColor(img3, cv2.COLOR_RGB2LAB) 
strColor=['L','A','B']
for j in range(3):  
    axs[j+i,0].imshow(p0[:,:,j],cmap='gray'),axs[j+i,0].set_title('LAB--'+strColor[j], fontsize=50)
    axs[j+i,1].imshow(p1[:,:,j],cmap='gray'),axs[j+i,1].set_title('LAB--'+strColor[j], fontsize=50)  
    axs[j+i,2].imshow(p2[:,:,j],cmap='gray'),axs[j+i,2].set_title('LAB--'+strColor[j], fontsize=50)  
    axs[j+i,3].imshow(p3[:,:,j],cmap='gray'),axs[j+i,3].set_title('LAB--'+strColor[j], fontsize=50)  
i=i+3 
p0 = cv2.cvtColor(img0, cv2.COLOR_RGB2YCrCb)  
p1 = cv2.cvtColor(img1, cv2.COLOR_RGB2YCrCb) 
p2 = cv2.cvtColor(img2, cv2.COLOR_RGB2YCrCb) 
p3 = cv2.cvtColor(img3, cv2.COLOR_RGB2YCrCb) 
strColor=['Y','Cr','Cb']
for j in range(3):  
	axs[j+i,0].imshow(p0[:,:,j],cmap='gray'),axs[j+i,0].set_title('YCrCb--'+strColor[j], fontsize=50) 
	axs[j+i,1].imshow(p1[:,:,j],cmap='gray'),axs[j+i,1].set_title('YCrCb--'+strColor[j], fontsize=50)   
	axs[j+i,2].imshow(p2[:,:,j],cmap='gray'),axs[j+i,2].set_title('YCrCb--'+strColor[j], fontsize=50)   
	axs[j+i,3].imshow(p3[:,:,j],cmap='gray'),axs[j+i,3].set_title('YCrCb--'+strColor[j], fontsize=50)   

plt.savefig('output_images/exploreColorspaces_LAB_YCrCb.jpg', bbox_inches='tight')

#-----------------------------------------
## explore HOG
#-----------------------------------------

_, hog_carY = get_hog_features(p0[:,:,0], orient=9,     pix_per_cell=8, cell_per_block=2, vis=True, feature_vec=False)
_, hog_carCr = get_hog_features(p0[:,:,1], orient=9,    pix_per_cell=8, cell_per_block=2, vis=True, feature_vec=False)
_, hog_carCb = get_hog_features(p0[:,:,2], orient=9,    pix_per_cell=8, cell_per_block=2, vis=True, feature_vec=False)
_, hog_noncarY =  get_hog_features(p3[:,:,0], orient=9, pix_per_cell=8, cell_per_block=2, vis=True, feature_vec=False)
_, hog_noncarCr = get_hog_features(p3[:,:,1], orient=9, pix_per_cell=8, cell_per_block=2, vis=True, feature_vec=False)
_, hog_noncarCb = get_hog_features(p3[:,:,2], orient=9, pix_per_cell=8, cell_per_block=2, vis=True, feature_vec=False)

f, axs = plt.subplots(4, 2, figsize=(10,20))
f.subplots_adjust(hspace = .4, wspace=.2)
i=0
axs[i,0].imshow(img0 ) ,axs[i,0].set_title('car',     fontsize=50)
axs[i,1].imshow(img3 ) ,axs[i,1].set_title('not car',     fontsize=50)
i=i+1
axs[i,0].imshow(hog_carY ) ,axs[i,0].set_title('car--Y',     fontsize=50)
axs[i,1].imshow(hog_noncarY ) ,axs[i,1].set_title('not car--Y',     fontsize=50)
i=i+1
axs[i,0].imshow(hog_carCr ) ,axs[i,0].set_title('car--Cr',     fontsize=50)
axs[i,1].imshow(hog_noncarCr ) ,axs[i,1].set_title('not car--Cr',     fontsize=50)
i=i+1
axs[i,0].imshow(hog_carCb ) ,axs[i,0].set_title('car--Cb',     fontsize=50)
axs[i,1].imshow(hog_noncarCb ) ,axs[i,1].set_title('not car--Cb',     fontsize=50)

plt.savefig('output_images/HOG_example.jpg', bbox_inches='tight')