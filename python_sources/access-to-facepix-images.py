#!/usr/bin/env python
# coding: utf-8

# **Access to FacePix dataset images:**<br/>
# The FacePix folder contains 5,430 face images related to 30 different persons. Each person has 181 different face images which captured in different pose angles from -90 to +90 degree (5,430 = 30 * 181). <br/>The file name format is as follows:
# /FacePix/P(D).jpg <br/>
# where P is the person index and can be in range(1, 30) and D is pose angle and can be in range (-90, +90). For Example /FacePix/20(-45).jpg represents the file name  of 20'th person's face image in pose -45 degrees.
# In the next code cell, the "get_image(person,pose)" method simply gets the person number and pose angle as inputs and returns the related face image.
# <br/><br/>
# ** All face images are cropped manually and resized to 60x51 and converted to GrayScale.**<br/>
# ** You can access to the original FacePix dataset here (and for more info):** https://cubic.asu.edu/content/facepix-database

# In[ ]:


import numpy as np # linear algebra
from skimage import io
from matplotlib import pyplot
import os

def get_item(person,pose):
    """ Returns an image of FacePix dataset.
    Inputs:
        person: Person number, in range(1,30).
        pose: Pose angle of face image, in range(-90,+90)"""
    
    facepix_path = "../input/facepix/FacePix/"
    img_path = facepix_path + "/" + str(person) + "("+ str(pose) +").jpg"
    img = io.imread(img_path)
    return img

# Reading 20'th person's face image in pose -45 degree
img = get_item(20,-45) 
pyplot.imshow(img,cmap='gray')
pyplot.show()

