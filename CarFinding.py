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

fname = 'test4.jpg' # try test1~6
img = mpimg.imread('./test_images/'+fname)
ystart = 400
ystop = 656
scale = 1.5
boxcolor=(0,0,255)    
#boxes, allboxes = find_cars(img, 
#                          ystart, 
#                          ystop, 
#                          scale, 
#                          svc, 
#                          X_scaler, 
#                          orient, 
#                          pix_per_cell, 
#                          cell_per_block, 
#                          boxcolor,
#                          spatial_size, 
#                          hist_bins)
#withWindows = draw_boxes(img, boxes, boxcolor, thick=6)
#plt.imshow(withWindows)
#plt.title('scale=1.5')
#plt.savefig('output_images/findCarsExample_'+fname, bbox_inches='tight')


# list seaching area
foundboxes=[]
for i in np.linspace(0,10,11):
    i = i.astype(int)
    ystart = 400
    ystop = 656
    scale = 1.5+i*0.1
    boxcolor = (i*0.1,i*0.1,1-i*0.1)
    boxcolor = tuple([255*x for x in boxcolor])
    boxes, allboxes = find_cars(img, 
                          ystart, 
                          ystop, 
                          scale, 
                          svc, 
                          X_scaler, 
                          orient, 
                          pix_per_cell, 
                          cell_per_block, 
                          boxcolor,
                          spatial_size, 
                          hist_bins)
    withWindows = draw_boxes(img, boxes, boxcolor, thick=6)
    withAllWindows = draw_boxes(img, allboxes, boxcolor, thick=6)
    foundboxes.append(boxes)
#    fig = plt.figure()
#    plt.subplot(121)
#    plt.imshow(withWindows)
#    plt.title('captured boxes')
#    plt.subplot(122)
#    plt.imshow(withAllWindows)
#    plt.title('all boxes')
#    plt.savefig('funPics/findingCars_'+str(i)+'.jpg', bbox_inches='tight')
    
#withWindows = np.copy(img)
#for i in range(len(foundboxes)):
#    boxcolor = (i*0.1,i*0.1,1-i*0.1)
#    boxcolor = tuple([255*x for x in boxcolor])
#    withWindows =draw_boxes(withWindows, foundboxes[i], boxcolor, thick=6)
#plt.imshow(withWindows)
#plt.savefig('output_images/findCarsMultiWin_'+fname, bbox_inches='tight')

for k in range(6):
    fname = 'test'+str(k+1)+'.jpg'
    img = mpimg.imread('./test_images/'+fname)
    foundboxes=[]
    foundAllboxes=[]
    for i in np.linspace(0,10,5):
        i = i.astype(int)
        ystart = 400
        ystop = 656
        scale = 1.5+i*0.1
        boxcolor = (i*0.1,i*0.1,1-i*0.1)
        boxcolor = tuple([255*x for x in boxcolor])
        boxes, allboxes = find_cars(img, 
                              ystart, 
                              ystop, 
                              scale, 
                              svc, 
                              X_scaler, 
                              orient, 
                              pix_per_cell, 
                              cell_per_block, 
                              boxcolor,
                              spatial_size, 
                              hist_bins)
        
#    withWindows = draw_boxes(img, boxes, boxcolor, thick=6)
#    withAllWindows = draw_boxes(img, allboxes, boxcolor, thick=6)
        foundboxes.append(boxes)
        foundAllboxes.append(allboxes)
    
    
#    fig = plt.figure(figsize=(20,15))
#    plt.subplot(121)
#    draw_image = np.copy(img)
# 
#    for j in range(len(foundboxes)):
#        draw_image = draw_boxes(draw_image, foundboxes[j], color=(0, 0, 255), thick=6)
#    plt.imshow(draw_image)
#   
#    plt.subplot(122)
#    draw_image = np.copy(img)
#    for j in range(len(foundAllboxes)):
#        draw_image = draw_boxes(draw_image, foundAllboxes[j], color=(0, 0, 255), thick=6)    
#    plt.imshow(draw_image)
#
#    plt.savefig('funPics/findingCars_fun'+str(k+1)+'.jpg', bbox_inches='tight')
    heatmap_img = np.zeros_like(img[:,:,0])
    for j in range(len(foundboxes)):
        heatmap_img = add_heat(heatmap_img, foundboxes[j])
    
    heatmap_img = apply_threshold(heatmap_img,1.5)
#    plt.figure(figsize=(10,10))
#    plt.imshow(heatmap_img, cmap='hot')
    labels = label(heatmap_img)
    plt.figure(figsize=(10,10))
    plt.imshow(labels[0], cmap='gray')
    print(labels[1], 'cars found')
    draw_image, rects = draw_labeled_bboxes(np.copy(img), labels)
    plt.imshow(draw_image)

    plt.savefig('funPics/findingCars_fun_heat'+str(k+1)+'.jpg', bbox_inches='tight')
###
