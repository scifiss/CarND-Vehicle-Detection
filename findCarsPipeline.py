# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 04:54:20 2017

@author: Rebecca
"""

import pickle
import cv2
from funLib import *
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label

data_pickle = pickle.load(open('./data_pickle.p','rb'))

X_scaler= data_pickle['X_scaler']  
mysvc=data_pickle['svc'] 


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

def process_image(img):
    foundboxes=[]
    foundAllboxes=[]
    for i in np.linspace(0,10,5):
        i = i.astype(int)
        ystart = 400
        ystop = 656
        scale = 1.5+i*0.1
 
        boxes, allboxes = find_cars(img, 
                              ystart, 
                              ystop, 
                              scale, 
                              mysvc, 
                              X_scaler, 
                              orient, 
                              pix_per_cell, 
                              cell_per_block, 
                              (0,0,255),
                              spatial_size, 
                              hist_bins)
        
#    withWindows = draw_boxes(img, boxes, boxcolor, thick=6)
#    withAllWindows = draw_boxes(img, allboxes, boxcolor, thick=6)
        foundboxes.append(boxes)
        foundAllboxes.append(allboxes)
    

    heatmap_img = np.zeros_like(img[:,:,0])
    for j in range(len(foundboxes)):
        heatmap_img = add_heat(heatmap_img, foundboxes[j])
    
    heatmap_img = apply_threshold(heatmap_img,1.5)
#    plt.figure(figsize=(10,10))
#    plt.imshow(heatmap_img, cmap='hot')
    labels = label(heatmap_img)
  
    draw_image, rects = draw_labeled_bboxes(np.copy(img), labels)
    return draw_image

def process_image1(img):
    boxes,allboxes = search_multiple_windows(img,mysvc,X_scaler,color_space, spatial_size, hist_bins, orient,pix_per_cell,
                            cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)
#    boxes,allboxes = search_multiple_windows(img)
   
    heatmap_img = np.zeros_like(img[:,:,0])
    heatmap_img = add_heat(heatmap_img, boxes)
    heatmap_img = apply_threshold(heatmap_img,1)
 
    labels = label(heatmap_img)

    draw_image, rects = draw_labeled_bboxes(np.copy(img), labels)

video_input = VideoFileClip('project_video.mp4')
video_clip = video_input.fl_image(process_image1)
video_clip.write_videofile('project_video_output1.mp4',audio=False)