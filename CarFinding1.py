# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 14:24:57 2017

@author: Rebecca
"""
import pickle
import cv2
from funLib import *
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
from scipy.ndimage.measurements import label

data_pickle = pickle.load(open('./data_pickle.p','rb'))

X_scaler= data_pickle['X_scaler']  
svc=data_pickle['svc'] 


color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
#hist_bins=32
#orient=9
# Define a single function that can extract features using hog sub-sampling and make predictions
# the code is basically from the course




###
# use slide_window() and search_windows() provided in the course
def search_multiple_windows(image):
    
    boxes = []
    allboxes = [] 
    for i in range(5):  #range(4): 
        y_start_stop=[375+i*10,375+300//(i+1)]   #[375+i*10,375+280//(i+1)]
        widwin = y_start_stop[1]-y_start_stop[0]
        windows = slide_window(image, x_start_stop=[None,None], y_start_stop=y_start_stop, 
                            xy_window=[widwin,widwin], xy_overlap=[0.75,0.75])
        
        allboxes += [windows]        
        
        boxes +=  search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)                       

    return boxes,allboxes

for i in range(6):
    fname = 'test'+str(i+1)+'.jpg'
    img = mpimg.imread('./test_images/'+fname)
    boxes,allboxes = search_multiple_windows(img)
    
    
    fig = plt.figure(figsize=(20,15))
    plt.subplot(121)
    draw_image = np.copy(img)
    draw_image = draw_boxes(draw_image, boxes, color=(0, 0, 255), thick=6)
    plt.imshow(draw_image)   
    plt.subplot(122)
    draw_image = np.copy(img)
    for j in range(len(allboxes)):
        draw_image = draw_boxes(draw_image, allboxes[j], color=(0, 0, 255), thick=6)    
    plt.imshow(draw_image)
    plt.savefig('funPics/findingCars_'+str(i+1)+'.jpg', bbox_inches='tight')
    heatmap_img = np.zeros_like(img[:,:,0])
    heatmap_img = add_heat(heatmap_img, boxes)
    heatmap_img = apply_threshold(heatmap_img,1)
    plt.figure(figsize=(10,10))
    plt.imshow(heatmap_img, cmap='hot')
    labels = label(heatmap_img)
    plt.figure(figsize=(10,10))
    plt.imshow(labels[0], cmap='gray')
    print(labels[1], 'cars found')
    draw_image, rects = draw_labeled_bboxes(np.copy(img), labels)
    plt.imshow(draw_image)

    plt.savefig('funPics/findingCars_heat'+str(i+1)+'.jpg', bbox_inches='tight')
    
images = glob.glob(('./videoCaptured/*.jpg'))
k=0
for imgfile in images:
    img = mpimg.imread(imgfile)
    boxes,allboxes = search_multiple_windows(img)
    
    
    fig = plt.figure(figsize=(20,15))
    plt.subplot(121)
    draw_image = np.copy(img)
    draw_image = draw_boxes(draw_image, boxes, color=(0, 0, 255), thick=6)
    plt.imshow(draw_image)
   
    plt.subplot(122)
    draw_image = np.copy(img)
    for j in range(len(allboxes)):
        draw_image = draw_boxes(draw_image, allboxes[j], color=(0, 0, 255), thick=6)    
    plt.imshow(draw_image)
    k=k+1
#
#    plt.savefig('funPics/CarsInVideo_'+str(k)+'.jpg', bbox_inches='tight')
    heatmap_img = np.zeros_like(img[:,:,0])
    heatmap_img = add_heat(heatmap_img, boxes)
    heatmap_img = apply_threshold(heatmap_img,1)
    #plt.figure(figsize=(10,10))
#    plt.imshow(heatmap_img, cmap='hot')
    labels = label(heatmap_img)
    plt.figure(figsize=(10,10))
#    plt.imshow(labels[0], cmap='gray')
    print(labels[1], 'cars found')
    draw_image, rects = draw_labeled_bboxes(np.copy(img), labels)
    plt.imshow(draw_image)
#    plt.set_title(labels[1]+ ' cars found')
    plt.savefig('funPics/CarsInVideo_heat'+str(k)+'.jpg', bbox_inches='tight')
    
