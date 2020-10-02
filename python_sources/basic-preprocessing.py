#!/usr/bin/env python
# coding: utf-8

# # **In this notebook I am going to do some basic data exploration.**

# *First of all let's figure out the size of the dataset that we have for polished analysis*

# In[ ]:


import pandas as pd
import numpy as np
import os
import shutil
import random

train_path = "../input/global-wheat-detection/train.csv"
images_path = "../input/global-wheat-detection/train/"

# We will load the csv data in a pandas dataframe and find its shape
data = pd.read_csv(train_path)
print(data.shape)

os.mkdir("un-annotated")

# We will count the number of images in the train folder
i = 0
for image in os.listdir(images_path):
    i += 1
print("Total number of images are %d", i)


# *As we can see that there is 43 times more data in the csv file that contains the bounding boxes for the wheat heads. This clearly states that each images has multiple bounding boxes.*
# 
# *Now lets have a look at the kind of data that we have in the csv file*

# In[ ]:


data.head()


# The file contains the width and the height of the image. The **bbox** column in the dataframe represents the dimensions of the bounding boxes in the format of [**xmin, ymin, width, height**]
# 
# The problem here is that the values in the bbox column is a string nor a list. We will have to convert it into a list.
# 
# But, an even more considerable problem is that, not all of the images in the train set have bounding boxes, but they donot affect the functioning of the model as they have no entries in the CSV file.

# # Splitting the data in the train and test set

# **In order to split the data in the train and the test sets, we take 100 unique image ids in the train csv and pick all the entries for that image id to move that into another cv file
# **
# 

# In[ ]:


test_ids = random.sample(list(data["image_id"].unique()), 650)
test = pd.DataFrame()
for id in test_ids:
  test = test.append(data[data['image_id'] == id])
  data.drop(data[data['image_id'] == id].index, inplace = True)


# Saving the test data file

# In[ ]:


test.to_csv("/kaggle/working/un-annotated/test.csv", index = False)


# ***Now its time to address the previously mentioned problem that states that the bbox data is of type string, we need it to be of type list and also we can change the format of the csv file, both train and test, to follow the genric format for object detection data files. This file then can also be directly consumed by tensorflow object detection API***

# Code for converting the train data file with columns [**image_id, height, width, xmin, ymin, xmax, ymax**]

# In[ ]:


values = []
filename = []

# Extracting the bbox values from all the rows, removing the brackets and converting them to list and finally appending the to another list
for index, row in data.iterrows():
  values.append(row.bbox.strip('[]').split(", "))
  filename.append(row.image_id + '.jpg')

# Type converting values from string to float
# Calculating xmax and ymax by adding width and height
# Saving the value in the corresponding list
xmin = []
ymin = []
xmax = []
ymax = []
for value in values:
  xmin.append(float(value[0]))
  ymin.append(float(value[1]))
  xmax.append(float(value[2]) + float(value[0]))
  ymax.append(float(value[3]) + float(value[1]))

# Preparing anew dataframe in the required format
processed_data = {}
processed_data["filename"] = filename
processed_data["width"] = data['width']
processed_data["height"] = data['height']
data['class'] = 'wheat'
processed_data["class"] = data['class']
processed_data["xmin"] = xmin
processed_data["ymin"] = ymin
processed_data["xmax"] = xmax
processed_data["ymax"] = ymax
processed_data = pd.DataFrame(processed_data)

# Saving the newly processed data in a file
processed_data.to_csv("/kaggle/working/un-annotated/train_processed.csv", index = False)


# ***Converting the test data in the required format***

# In[ ]:


test = pd.read_csv("/kaggle/working/un-annotated/test.csv")
values = []
filename = []

# Extracting the bbox values from all the rows, removing the brackets and converting them to list and finally appending the to another list
for index, row in data.iterrows():
  values.append(row.bbox.strip('[]').split(", "))
  filename.append(row.image_id + '.jpg')

# Type converting values from string to float
# Calculating xmax and ymax by adding width and height
# Saving the value in the corresponding list
xmin = []
ymin = []
xmax = []
ymax = []
for value in values:
  xmin.append(float(value[0]))
  ymin.append(float(value[1]))
  xmax.append(float(value[2]) + float(value[0]))
  ymax.append(float(value[3]) + float(value[1]))

# Preparing anew dataframe in the required format
processed_data = {}
processed_data["filename"] = filename
processed_data["width"] = data['width']
processed_data["height"] = data['height']
data['class'] = 'wheat'
processed_data["class"] = data['class']
processed_data["xmin"] = xmin
processed_data["ymin"] = ymin
processed_data["xmax"] = xmax
processed_data["ymax"] = ymax
processed_data = pd.DataFrame(processed_data)

# Saving the newly processed data in a file
processed_data.to_csv("/kaggle/working/un-annotated/test_processed.csv", index = False)

