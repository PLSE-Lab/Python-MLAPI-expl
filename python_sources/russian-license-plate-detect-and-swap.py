#!/usr/bin/env python
# coding: utf-8

# **OBJECTIVE:** The objective of this analysis is to use object detection in OpenCV with Python bindings to:
# 1. detect Russian license plates in an image containing 2 Russian plates, and
# 2. swap one of the plates with another plate in the image
# 
# Object detection will be performed using Haar feature-based cascade classifiers. This will be a very simple machine learning approach to object detection in OpenCV. To learn how Haar cascade object detection works before you begin this notebook, check out this site: https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html.

# To begin, we will need the following imports.

# In[ ]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Next, read in an image containing 2 Russian license plates using OpenCV's *imread()*. Then display the image using matplotlib.
# 
# I found this image on the web: https://www.themoscowtimes.com/2015/10/09/russian-recall-of-lucky-thief-license-plates-following-disruptive-popularity-a50175.

# In[ ]:


img = cv2.imread('../input/russian_plate.jpg')
plt.imshow(img)


# Uhg! The colors of our image are not quite right. Why is this? This is because we read our image in using OpenCV. OpenCV represents RGB images in reverse order. This means that images are actually represented in BGR order rather than RGB. If we want to display our OpenCV image using matplotlib, we need to convert it from BGR to RGB.

# In[ ]:


#I will make a bit larger so you can compare this with the final swapped plate image
fig, ax = plt.subplots(figsize=(20,20))
ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# Nice! To detect the Russian license plates in this image, I will use a pretrained cascade classifier rather than train my own.  I obtained this pretrained classifier *xml* file from https://github.com/opencv/opencv/tree/master/data/haarcascades. There are many pretrained classifier *xml* files included on this site for detecting objects of a particular type, e.g. faces (frontal, profile), pedestrians etc. I will chose the one that works best for detecting Russian license plates: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_russian_plate_number.xml.

# In[ ]:


pretrained = '../input/haarcascade_russian_plate_number.xml'


# Object detection works best on grayscale images.  So, I will convert my image to grayscale. Next, I will create an OpenCV trained classifier object using the pretrained classifier *xml* file referenced above. Then I will use it to detect the Russian plates in the grayscale image by applying OpenCV's *detectMultiScale()* function. For a more detailed explanation regarding *detectMultiScale()* parameters, check out this site: https://stackoverflow.com/questions/51132674/meaning-of-parameters-of-detectmultiscalea-b-c

# In[ ]:


#converting the image to grayscale is needed for detection by the classifier
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#create a trained classifier object with the xml file
plate_cascade = cv2.CascadeClassifier(pretrained)

#detect the plate with the classifier
plate = plate_cascade.detectMultiScale(gray, scaleFactor = 1.05, minNeighbors = 5, minSize = (40,40))
    
#opting to print bounding boxes to console; this is useful for the plates defined later on -- will hard code this manually
print(plate)


# Once the dection is done, the result is a list of bounding boxes (*x, y, w, h*) representing the location of each plate the classifier found. Unfortunately, each time you run the above code, the order of the plates in the image that are detected will be different.  So, I will have to do a bit of hard coding (you will see later) to ensure that the coordinates for the correct plates are referenced with each run. 
# 
# But wait! There are only 2 plates in the image. Why are there 4 bounding box items in our listing? Unfortunately, our classifier is detecting things that are clearly not plates.  This could be due to a number of reasons: the cascade classifier xml file may not be the best, the image itself may need to be preprocessed to yield better results, the parameters chosen for the *detectMultScale()* function may need to be adjusted.  I will leave this up to you.

# Let's take a look at the plates our classifier detected by drawing red bounding boxes on our image using the location coordinate results.

# In[ ]:


#use count to keep track of which are plates I want to swap
#the order of the plates detected is random each time you run this notebook; this is why I will manually hard code this later on
count = 0

#draw a rectangle around each detected plate with the list of returned bounding box coordinates
#don't want to overwrite the original image
detected_img = img

for (x, y, w, h) in plate:
    cv2.rectangle(detected_img, (x,y), (x + w, y + h), (0,0,255), 2) #red bounding box
    cv2.putText(detected_img, str(count), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    count+=1
    
#let's see what plates were detected; remember to reverse the color order
plt.imshow(cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB))


# Yes, it is now clear we have detected 2 additional objects in the image that are not plates. I am opting to proceed without improving my results.  
# 
# Next, I will do a bit of hard coding so that I am referencing the correct plates with each new run: the plate I want to swap (*swap_this_plate_coords*), the plate I am swapping this one with (*with_this_plate_coords*). You may think of a better way to do this.

# In[ ]:


#plate of car on left has coordinates [ 291  435  133   45]
#plate of car on right has coordintes [ 875  414  137   46]

#here is the hard coding I mentioned previously
swap_this_plate_coords = [875,414,137,46]
with_this_plate_coords = [291,435,133,45]


# Keep in mind that if you are using your own image, your coordinates will be different from mine. You will need to identify the coordinates for each corresponding plate in your analysis. 
# 
# 
# Notice here that the *h* and *w* values for each of my plates are different.  This means that they have different height and width dimensions.  Before swapping, I will resize the smaller plate (*with_this_plate_coords*) so that it is the same size as the other (*swap_this_plate_coords*). This will make for an easier swap.

# In[ ]:


#don't want to overwrite the original image; create a new image
img = cv2.imread('../input/russian_plate.jpg')
swap_img = img

#dims will be the width and height of swap_this_plate
dims = (137,46)

#slice out the part of the image that is with_this_plate; hard coded using with_this_plate_coords
#resize with_this_plate to be the same size as swap_this_plate (= dims); swap_this_plate has slightly larger dims
with_this_plate = img[with_this_plate_coords[1]:(with_this_plate_coords[1] + with_this_plate_coords[3]), with_this_plate_coords[0]:(with_this_plate_coords[0] + with_this_plate_coords[2]), :]
with_this_plate = cv2.resize(with_this_plate, dims, interpolation = cv2.INTER_AREA)

#these are swap_this_plate pixel areas to replace; hard coded using swap_this_plate_coords
for i,x in enumerate(range(swap_this_plate_coords[1],(swap_this_plate_coords[1] + swap_this_plate_coords[3]))):
    for j,y in enumerate(range(swap_this_plate_coords[0],(swap_this_plate_coords[0] + swap_this_plate_coords[2]))):
        swap_img[x,y,:] = with_this_plate[i,j,:]


# Whew! First, I sliced out the part of the image that is *with_this_plate_coords* using array slicing and the location coordinates -- saved this part as *with_this_plate*.  Then I resized this portion as described previously.  Next, I replaced every pixel from the part of the image that is *swap_this_plate_coords* with that of *with_this_plate*.
# 
# Let's display the image to see if it worked! I will put in some text to make things more clear.

# In[ ]:


#put some text to identify which plate was swapped
cv2.putText(swap_img, 'SWAPPED!', (swap_this_plate_coords[0],swap_this_plate_coords[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

#let's see our image with the plate swapped
#I'll make the image a bit larger so you can see
fig, ax = plt.subplots(figsize=(20,20))
ax.imshow(cv2.cvtColor(swap_img, cv2.COLOR_BGR2RGB))


# It worked! Look at the license plates of the 2 cars in the image. If I did not put in the red text, someone would think both of these cars had the same license plates. This shows us one way to use Computer Vision to create faked content.
# 
# This was a lot of fun.  Try repeating this analysis using your own image. Feel free to make improvements!
