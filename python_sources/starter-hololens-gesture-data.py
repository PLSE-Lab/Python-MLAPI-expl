#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import math


# # Introduction
# 
# Welcome to the HoloLens Gesture recognition dataset which was created during my Bachelor Thesis. The goal of the thesis was to classify between 4 different hand gestures done on the Microsoft HoloLens 2 using convolutional neural networks. The different gestures are a line, circle, wave and a V. The HoloLens was used to capture the hand drawing those gestures into the air.
# 
# This Notebook contains some code to help you getting started with the HoloLens gesture data set. For simplicity reasons the code in this notebook only handles two dimensional Data. 
# 
# **This Means:** Even though the dataset contains a lists of (X,Y,Z) coordinates of the hand performing gestures, only (X,Y) coordinate pairs will be handled here.

# # Exploratory Analysis
# 
# Lets first have a look on the raw data. In the following example you can find a 2D plot of the raw data of a circle gesture. You can simply use this code to explore the other gestures (line, V and wave) as well. Just import the gestures from the other `.json` files to do so.

# In[ ]:


with open('/kaggle/input/circle_gestures.json') as data_file:    
    circle_gestures = json.load(data_file)

#change index to plot other data in the list.
index_to_show = 0
    
sample = circle_gestures[index_to_show]['gestureData']
x = [coord['x'] for coord in sample]
y = [coord['y'] for coord in sample]

plt.scatter(x,y)
plt.show()


# # Data preprocessing
# 
# The first and most important step I took in data preprocessing to discretize the raw data. 
# The raw data is a list of sequential (x,y,z) coordinate pairs. The goal is to map those coordinate pairs onto a discrete scale.
# 
# Below you can find the algorithm:

# In[ ]:


INF = 999999
NEGINF = -999999

def discretize_2D(gesture_list):
    
    PADDING_FACTOR = 5
    cube_x = 29
    cube_y = 29
    
    #cube size is cube_y + 2*PADDING_FACTOR + 1!!!
    
    cubes = []
    for i in gesture_list:
        gestureData = i['gestureData']
        gestureRange = GestureRange()
        #get maximum and minimum x & y values
        for j in gestureData:
            if(j['x'] > gestureRange.x_max):
                gestureRange.x_max = j['x']
            if(j['x'] < gestureRange.x_min):
                gestureRange.x_min = j['x']
            
            if(j['y'] > gestureRange.y_max):
                gestureRange.y_max = j['y']
            if(j['y'] < gestureRange.y_min):
                gestureRange.y_min = j['y']
                
        #get range of x and y data
        x_range = gestureRange.x_max - gestureRange.x_min
        y_range = gestureRange.y_max - gestureRange.y_min
        
        #get factor for discretization. Itution: How many X or Y Values fit in one discretized "step"
        discretization_factor_x  = x_range/cube_x
        discretization_factor_y = y_range/cube_y
                
        #The bigger factor counts.
        step_max  = max([discretization_factor_x, discretization_factor_y])
        
        #Set data to all 0
        cubeData = np.zeros((cube_x+1,cube_y+1))
           
        frame_count = 0
        for j in gestureData:
            #zero center data
            x_zeroed = j['x'] - gestureRange.x_min	
            y_zeroed = j['y'] - gestureRange.y_min
            
            #get discretized value
            x_discretized = int(x_zeroed / step_max)      
            y_discretized = int(y_zeroed / step_max)
            
            frame_count = frame_count +1
            cubeData[x_discretized,y_discretized] = 1
            
            
        #create padding with zeros
        padded_cube_data = np.lib.pad(cubeData, PADDING_FACTOR, 'minimum')
        
        #wire it all together
        original_cube = {}
        original_cube['gestureData'] = padded_cube_data
        original_cube['gestureClass'] = i['gestureClass']
        
        cubes.append(original_cube)
      
    return cubes

class GestureRange:
    x_min = INF
    x_max = NEGINF
    y_min = INF
    y_max = NEGINF
    z_min = INF
    z_max = NEGINF


# ... And a plot of a discretized 2D gesture

# In[ ]:


discrete_circle_gestures = discretize_2D(circle_gestures)
sample_discrete_circle_gesture = discrete_circle_gestures[0]

plt.imshow(sample_discrete_circle_gesture['gestureData'])


# ## Your turn
# 
# using the discretization method we reducing the variety of the input features. Doing so enables us to effectively apply classification alorithms on the data. In my Bachelor thesis I used a CNN for classification.
# 
# Have fun :)
