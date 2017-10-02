**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image0]: ./output_images/exampleCarNNoncar.jpg

[image1]: ./output_images/exploreColorspaces_RGB_HSV.jpg
[image2]: ./output_images/exploreColorspaces_LAB_YCrCb.jpg
[image3]: ./output_images/HOG_example.jpg
[image4]: ./output_images/HOG_tryPara.jpg
[image5]: ./output_images/test1_marked.jpg
[image6]: ./output_images/findingCars_1.jpg
[image7]: ./output_images/findingCars_3.jpg
[image8]: ./output_images/findingCars_fun1.jpg
[image9]: ./output_images/findingCars_fun3.jpg
[image14]: ./output_images/findingCars_fun_heat1.jpg
[image15]: ./output_images/findingCars_fun_heat4.jpg
[image10]: ./output_images/CarsInVideo_heat0.jpg
[image11]: ./output_images/CarsInVideo_heat7.jpg
[image12]: ./output_images/CarsInVideo_heat25.jpg
[image13]: ./output_images/CarsInVideo_heat38.jpg
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  
You're reading it!

###Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images. There are 8792  car images and 8968  non-car .png images of size(64X64X3) in the dataset. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image0]

I grabbed random images from each of the two classes and displayed them to get a feel for different colorspaces.
Here is an example using the 'RGB' and `HSV` color space 

![alt text][image1]

Here is an example using the 'LAB' and `YCrCb` color space 

![alt text][image2]

I then tried the skimage.hog() with HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`

![alt text][image3]
As is seen, LAB-L, all RGB, HSV-V, YCrCb-Y provide good representations to most car structures.

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters, with pixels_per_cell from 8 to 16 and cells_per_block from 2 to 4 on Y channel of YCrCb colorspace. 
![alt text][image4]
I find `orientations=9`, `pixels_per_cell` at a size of no bigger than `(8, 8)`  can conserve the spatial structure of a car. 

pixels_per_cell seems not a sensitive parameter to the extracted feature, but the number has a great impact to the final feature vector size. `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` gives (7,7,2,2,9) and `pixels_per_cell=(8, 8)` and `cells_per_block=(4, 4)` gives (5,5,4,4,9). 

Finally I choose `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`.
When choosing the colorspace to do HOG, colorspaces (for example, YUV) can't be used since it contains negative values.  I use YCrCb colorspace because for non car images, the 3 channels usually differ a lot, so provide more information.

The trying of features are coded in 'featureExtraction.py'.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using C=0.1,1, and 10,with all channels from HOG features. It turns out all Cs produces a same accuracy (0.989302). The code is between line 24 and 40 in 'classifierTraiig.py'. I also tried `rbf` as the kernel, which gives a higher accuracy (0.991). 

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The searching window should move in the image, changing both positions and sizes in the area where a car could appear (y<=656 and y>=400).
 
![alt text][image5]

The first approach is to try windows with different starting positions and sizes, i.e. bigger when the window bottom is close to the image bottom (the car is bigger when being closer). The code is based from the course, and implemented in `CarFinding1.py` (line 42~61). 

![alt text][image6]  
![alt text][image7]

Another approach is to use scales to change the window sizes, implemented in `CarFinding.py`, using function `find_cars()` in funLib.py. There are 11 scales from 1.5 to 2.5. Results for same images are shown here.
![alt text][image8]  
![alt text][image9]
Many trials and errors went through with getting a proper scale for windows searching. The script is robust unless a car is cut in the image.
After this, a heat map is applied to get overlapped windows, and a threshold is applied to select only the area of heat higher than the threshold.
```
    heatmap_img = np.zeros_like(img[:,:,0])
    heatmap_img = add_heat(heatmap_img, boxes)
    heatmap_img = apply_threshold(heatmap_img,1) 
    labels = label(heatmap_img)
    draw_image, rects = draw_labeled_bboxes(np.copy(img), labels)
```

The 2nd approach offers more densly windows, so the threshold to the heat map is higher than the 1st approach.
####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. They are implemented between line 55 and 107 in 'funLib.py'. Here are some example images:

![alt text][image14]
![alt text][image15]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)  Also at Youtube (https://youtu.be/rS4u6strDRI) (approach 2) and (https://youtu.be/cD_tOnOopN8) (approach 1)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here the resulting bounding boxes are drawn onto the last frame in the series:

![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
More results are saved in (./funPics)
---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

- I think the threshold to heatmap should be dinamically adjusted. The near is the car, the higher the threshold, since there are many windows overlapped. The further the car, the lower the threshold. The small car in the far area might only get one window detected.

- I spent a lengthy time trying to find the sliding windows. Again, the windows should cover the area where a car could appear, and also should vary its size since a passing car appears bigger when being closer. If distortion and perspective transformation are done, the windows would not need to change sizes.

- If obtained boxes of previous images are recorded, the sudden disppearing of boxes in some frames could be removed, like what I did in Project 4.
