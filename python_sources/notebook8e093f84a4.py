#!/usr/bin/env python
# coding: utf-8

# # Import Library

# In[ ]:


import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import skimage.feature
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
get_ipython().run_line_magic('matplotlib', 'inline')

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))


# # File Extract & Sort

# In[ ]:


file_names = os.listdir("../input/Train/")
file_names = sorted(file_names, key=lambda 
                    item: (int(item.partition('.')[0]) if item[0].isdigit() else float('inf'), item))

# select a subset of files to run on
file_names = file_names[0:1]


# #  Extract Coordinate of Dotted SeaLion

# In[ ]:


# dataframe to store coordinate results in
classes = ["adult_males", "subadult_males", "adult_females", "juveniles", "pups"]
coordinates_df = pd.DataFrame(index=file_names, columns=classes)


# In[ ]:


def get_coordinate(filename):
    
    # read the Train and Train Dotted images
    image_1 = cv2.imread("../input/TrainDotted/" + filename)
    image_2 = cv2.imread("../input/Train/" + filename)
    
    # absolute difference between Train and Train Dotted
    image_3 = cv2.absdiff(image_1,image_2)
    
    # mask out blackened regions from Train Dotted
    mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    mask_1[mask_1 < 20] = 0
    mask_1[mask_1 > 0] = 255
    
    mask_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    mask_2[mask_2 < 20] = 0
    mask_2[mask_2 > 0] = 255
    
    image_3 = cv2.bitwise_or(image_3, image_3, mask=mask_1)
    image_3 = cv2.bitwise_or(image_3, image_3, mask=mask_2) 
    
    # convert to grayscale to be accepted by skimage.feature.blob_log
    image_3 = cv2.cvtColor(image_3, cv2.COLOR_BGR2GRAY)
    
    # detect blobs
    blobs = skimage.feature.blob_log(image_3, min_sigma=3, max_sigma=4, num_sigma=1, threshold=0.02)
    
    adult_males = []
    subadult_males = []
    pups = []
    juveniles = []
    adult_females = [] 
    
    for blob in blobs:
        # get the coordinates for each blob
        y, x, s = blob
        # get the color of the pixel from Train Dotted in the center of the blob
        g,b,r = image_1[int(y)][int(x)][:]
        
        # decision tree to pick the class of the blob by looking at the color in Train Dotted
        if r > 200 and g < 50 and b < 50: # RED
            adult_males.append((int(x),int(y)))        
        elif r > 200 and g > 200 and b < 50: # MAGENTA
            subadult_males.append((int(x),int(y)))         
        elif r < 100 and g < 100 and 150 < b < 200: # GREEN
            pups.append((int(x),int(y)))
        elif r < 100 and  100 < g and b < 100: # BLUE
            juveniles.append((int(x),int(y))) 
        elif r < 150 and g < 50 and b < 100:  # BROWN
            adult_females.append((int(x),int(y)))
            
    coordinates_df["adult_males"][filename] = adult_males
    coordinates_df["subadult_males"][filename] = subadult_males
    coordinates_df["adult_females"][filename] = adult_females
    coordinates_df["juveniles"][filename] = juveniles
    coordinates_df["pups"][filename] = pups
    
    return coordinates_df


# In[ ]:


def get_blobs(filename):
    # read the Train and Train Dotted images
    image_1 = cv2.imread("../input/TrainDotted/" + filename)
    image_2 = cv2.imread("../input/Train/" + filename)
    
    # absolute difference between Train and Train Dotted
    image_3 = cv2.absdiff(image_1,image_2)
    
    # mask out blackened regions from Train Dotted
    mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    mask_1[mask_1 < 20] = 0
    mask_1[mask_1 > 0] = 255
    
    mask_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    mask_2[mask_2 < 20] = 0
    mask_2[mask_2 > 0] = 255
    
    image_3 = cv2.bitwise_or(image_3, image_3, mask=mask_1)
    image_3 = cv2.bitwise_or(image_3, image_3, mask=mask_2) 
    
    # convert to grayscale to be accepted by skimage.feature.blob_log
    image_3 = cv2.cvtColor(image_3, cv2.COLOR_BGR2GRAY)
    
    # detect blobs
    blobs = skimage.feature.blob_log(image_3, min_sigma=3, max_sigma=4, num_sigma=1, threshold=0.02)
    
    return blobs


# In[ ]:


def get_xy_range_basic(size):
    ### return (x_left, x_right, y_up, y_down)
    return (size,size,size,size) 


# In[ ]:


def parse_image(filename):
    ### get coordinate of all sea lions
    Dict_range = {}
    blobs = get_blobs(filename)
    
    for blob in blobs:
        # get the coordinates for each blob
        y, x, s = blob
        
        xy_range = get_xy_range_basic(size=16)
        Dict_range[(x,y)] = xy_range
    
    ### output sub_image and annotation file for each blob
    for key in Dict_range.keys():
        if(key in Dict_range):
            main_x = key[0]
            main_y = key[1]
            xy_range = Dict_range[key]
            x_min = main_x - xy_range[0]
            x_max = main_x + xy_range[1]
            y_min = main_y - xy_range[2]
            y_max = main_y + xy_range[3]
            
            ### get basic sub_image
            
            
            if()
        
        


# # Display Extracted Fixed Size Square Image 

# In[ ]:


#HALF_SQUARE_SIZE = 16
x = []
y = []

for filename in file_names:
    coordinates_df = get_coordinate(filename)
    image = cv2.imread("../input/Train/" + filename)
    for lion_class in classes:
        for coordinates in coordinates_df[lion_class][filename]:
            thumb = image[coordinates[1]-16:coordinates[1]+16,coordinates[0]-16:coordinates[0]+16,:]
            if np.shape(thumb) == (32, 32, 3):
                x.append(thumb)
                y.append(lion_class)
x = np.array(x)
y = np.array(y)


# In[ ]:


for lion_class in classes:
    f, ax = plt.subplots(1,10,figsize=(12,1.5))
    f.suptitle(lion_class)
    axes = ax.flatten()
    j = 0
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
        for i in range(j,len(x)):
            if y[i] == lion_class:
                j = i+1
                a.imshow(cv2.cvtColor(x[i], cv2.COLOR_BGR2RGB))
                break


# Thanks to Radu Stoicescu's kernel - "Use keras to classify Sea Lions: 0.91 accuracy"
# https://www.kaggle.com/radustoicescu/noaa-fisheries-steller-sea-lion-population-count/use-keras-to-classify-sea-lions-0-91-accuracy

# # Output annotation file

# In[ ]:


# class_id  x_pos/image_x_width  y_pos/image_y_height  x_len/image_x_width  y_len/image_y_height
def output_annotation(filename):
    coordinates_df = get_coordinate(filename)
    print(coordinates_df)
    txt_outfile = open("{0}_anno.txt".format(filename), "w")  #./annotations/
    
    image = cv2.imread("../input/Train/" + filename)
    image_W = float(image.shape[1])
    image_H = float(image.shape[0])    
    
    for idx in range(len(classes)):
        for coordinates in coordinates_df[classes[idx]][filename]:
            Width = 32   # int
            Height = 32  # int
            
            x_pos = float(coordinates[0])/image_W
            y_pos = float(coordinates[1])/image_H
            x_len = float(Width)/image_W
            y_len = float(Height)/image_H
            element = [x_pos, y_pos, x_len, y_len]
                       
            txt_outfile.write(str(idx) + " " + " ".join([str(x) for x in element]) + '\n')
     
    txt_outfile.close()


# In[ ]:


output_annotation(file_names[0])


# In[ ]:


image = cv2.imread("../input/Train/" + file_names[0])
image.shape


# In[ ]:


plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


# In[ ]:


File = open("{0}_anno.txt".format(file_names[0]), "r")
File.read()


# In[ ]:





# In[ ]:


file_names

