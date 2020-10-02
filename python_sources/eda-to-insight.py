#!/usr/bin/env python
# coding: utf-8

# ### Data Description :
# Much of the Text was taken from the official page 
# This dataset contains photos of streets, taken from the roof of a car. We're attempting to predict the position and orientation of all un-masked cars in the test images. You should also provide a confidence score indicating how sure you are of your prediction.
# 
# Pose Information (train.csv) Note that rotation values are angles expressed in radians, relative to the camera.
# 
# The primary data is images of cars and related pose information. The pose information is formatted as strings, as follows: model type, yaw, pitch, roll, x, y, z
# 
# A concrete example with two cars in the photo: 5 0.5 0.5 0.5 0.0 0.0 0.0 32 0.25 0.25 0.25 0.5 0.4 0.7
# 
# Submissions (per sample_submission.csv) are very similar, with the addition of a confidence score, and the removal of the model type. You are not required to predict the model type of the vehicle in question.
# 
# ID, PredictionString ID_1d7bc9b31,0.5 0.5 0.5 0.0 0.0 0.0 1.0 indicating that this prediction has a confidence score of 1.0.
# 
# Other Data:
# 
# Image Masks (test_masks.zip / train_masks.zip) Some cars in the images are not of interest (too far away, etc.). Binary masks are provided to allow competitors to remove them from consideration.
# 
# Car Models 3D models of all cars of interest are available for download as pickle files - they can be compared against cars in images, used as references for rotation, etc.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage import io
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# Any results you write to the current directory are saved as output.


# In[ ]:


trainData =  pd.read_csv("../input/pku-autonomous-driving/train.csv")
trainData.head()


# In above dataframe trainData, let us visualize the first five image. We are having all the image in train_images.zip directory.

# ### How many image do we have in train data ?

# In[ ]:


trainData.shape


# In[ ]:


trainpath = "../input/pku-autonomous-driving/train_images/" 


# ### Image ID_8a6e65317.jpg

# In[ ]:


img0 = io.imread(trainpath+"ID_8a6e65317.jpg")
io.imshow(img0)


# ### Image ID_337ddc495.jpg

# In[ ]:


img1 = io.imread(trainpath+"ID_337ddc495.jpg")
io.imshow(img1)


# 

# In[ ]:


img2 = io.imread(trainpath+"ID_a381bf4d0.jpg")
io.imshow(img2)


# 

# In[ ]:


img3 = io.imread(trainpath+"ID_7c4a3e0aa.jpg")
io.imshow(img3)


# ### What is inside masks 

# In[ ]:


maskpath = "../input/pku-autonomous-driving/train_masks/"


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


img0 = io.imread(trainpath+"ID_8a6e65317.jpg")
fig = plt.figure(figsize=(20,10))

ax1 = fig.add_subplot(1,2,1)
ax1.imshow(img0)
img0m = io.imread(maskpath+"ID_8a6e65317.jpg")
ax2 = fig.add_subplot(1,2,2)
ax2.imshow(img0m)


# ### For ID_8a6e65317.jpg How many car are in picture.

# In[ ]:


trainData.PredictionString[0]


# In[ ]:


### Every string has data of more than vehicles in Image ID_8a6e65317.jpg. 
numOfCars = len(trainData.PredictionString[0].split(" "))/7
print("Number of cars in image ID_8a6e65317.jpg is : ", numOfCars)


# It seems that, list of dictionary will be best to possecces this sort of data as follows.

# In[ ]:


data = trainData.PredictionString[0]
dataList = data.split(" ")
numOfVehicles = len(dataList) /7
variables = ["modeltype", "yaw", "pitch", "roll", "x", "y", "z"]
listOfData = []
for i in range(0,len(dataList),7) :
    
    lastIndex = i+7
    dt = dataList[i:lastIndex:1]
    dct = dict(zip(variables,dt))
    listOfData.append(dct)
    
    


# In[ ]:


listOfData


# Let us do the same analysis for second image data in list 

# In[ ]:


img1 = io.imread(trainpath+"ID_337ddc495.jpg")
fig = plt.figure(figsize=(20,10))

ax1 = fig.add_subplot(1,2,1)
ax1.imshow(img0)
img1m = io.imread(maskpath+"ID_337ddc495.jpg")
ax2 = fig.add_subplot(1,2,2)
ax2.imshow(img0m)


# In[ ]:


img1m.shape


# ### All layers in mask image has same data. That we can visualize from following code line

# In[ ]:


(img1m[0,:,:] == img1m[1,:,:]).all()


# In[ ]:


numOfCars = len(trainData.PredictionString[1].split(" "))/7
print("Number of cars in image ID_337ddc495.jpg is : ", numOfCars)


# ### Let us create a new column in our dataframe trainData, which consists of dictionaries of vehicle data.

# In[ ]:


data = trainData.PredictionString[0]
def addDictionary(data) :

    dataList = data.split(" ")
    numOfVehicles = len(dataList) /7
    variables = ["modeltype", "yaw", "pitch", "roll", "x", "y", "z"]
    listOfData = []
    for i in range(0,len(dataList),7) :
    
        lastIndex = i+7
        dt = dataList[i:lastIndex:1]
        dct = dict(zip(variables,dt))
        listOfData.append(dct)
    return listOfData
addDictionary(data)


# In[ ]:


trainData["dictVal"] = trainData.PredictionString.apply(lambda x : addDictionary(x))


# In[ ]:


trainData.head()


# ### Calculating number of vehicles in each pictures.

# In[ ]:


trainData["noOfVehicle"] = trainData.dictVal.apply(lambda x : len(x))


# In[ ]:


trainData.head()


# ### What is average number of vehicles in each picture 

# In[ ]:


print("Average number of vehicles in pictures : ",trainData["noOfVehicle"].mean())


# # To be continued ....

# # Kindly upvote if you like it :)

# In[ ]:




