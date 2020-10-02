#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# From [Face Detection using OpenCV](https://www.kaggle.com/serkanpeldek/face-detection-with-opencv) by @serkanpeldek we got and slightly modified the functions to extract the cat face.
# 

# # Data preparation
# 
# ## Load packages  
# 
# We start by loading the packages.

# In[ ]:


import numpy as np
import pandas as pd
import os
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2 as cv


# ## Load Haar Cascade resource
# 
# Here we load the Haar Cascade resource for Cat Face detection.

# In[ ]:


FACE_DETECTION_FOLDER = "..//input//cat-face-detection//"
#Frontal cat face detector
frontal_cascade_path = os.path.join(FACE_DETECTION_FOLDER,'haarcascade_frontalcatface.xml')


# # Cat Face Detection
# 
# The class CatFaceDetector initialize the cascade classifier (using the imported resource for cat face detection). The function detect uses a method of the CascadeClassifier to detect objects into images - in this case the cat face.

# In[ ]:


class CatFaceDetector():
    '''
    Class for Cat Face Detection
    '''
    def __init__(self,object_cascade_path):
        '''
        param: object_cascade_path - path for the *.xml defining the parameters for cat face detection algorithm
        source of the haarcascade resource is: https://github.com/opencv/opencv/tree/master/data/haarcascades
        '''

        self.objectCascade=cv.CascadeClassifier(object_cascade_path)


    def detect(self, image, scale_factor=1.15,
               min_neighbors=1,
               min_size=(30,30)):
        '''
        Function return rectangle coordinates of cat face for given image
        param: image - image to process
        param: scale_factor - scale factor used for cat face detection
        param: min_neighbors - minimum number of parameters considered during cat face detection
        param: min_size - minimum size of bounding box for object detected
        '''
        bbox=self.objectCascade.detectMultiScale(image,
                                                scaleFactor=scale_factor,
                                                minNeighbors=min_neighbors,
                                                minSize=min_size)
        return bbox


# We initialize an object of type CatFaceDetector.

# In[ ]:


#Detector for cat frontal face detectiob created
fcfd=CatFaceDetector(frontal_cascade_path)


# We define a function for cat face detection. The detected cat face is marked with an orange circle.

# In[ ]:


def detect_cat_face(image, scale_factor, min_neighbors, min_size):
    '''
    Cat Face detection function
    Identify frontal cat face and display the detected marker over the image
    param: image - the image extracted from the video
    param: scale_factor - scale factor parameter for `detect` function of ObjectDetector object
    param: min_neighbors - min neighbors parameter for `detect` function of ObjectDetector object
    param: min_size - minimum size parameter for f`detect` function of ObjectDetector object
    '''
    
    image_gray=cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    cat_face=fcfd.detect(image_gray,
                   scale_factor=scale_factor,
                   min_neighbors=min_neighbors,
                   min_size=min_size)

    for x, y, w, h in cat_face:
        #detected cat face shown in color image
        cv.circle(image,(int(x+w/2),int(y+h/2)),(int((w + h)/4)),(0, 127,255),3)
        #cv.rectangle(image,(x,y),(x+w, y+h),(255, 127,0),3)

    # image
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    ax.imshow(image)


# We import a cat image and we use the Cat Face Detector function (**cat_face_detect**) to detect the cat face.

# In[ ]:


# cat image
img_source = cv.imread(os.path.join(FACE_DETECTION_FOLDER,"cat.jpg"))
# detect face and show cat face marker over image
detect_cat_face(image=img_source,scale_factor=1.15, min_neighbors=1, min_size=(30, 30)) 


# # More inspiration
# 
# Use these resources to start tagging your favorite cat pictures.   
# 
# Have fun!
