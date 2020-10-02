#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import random
from skimage import io
from shutil import copyfile
import sys
import time

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# # Load the datasets

# In[ ]:


train = pd.read_csv("/kaggle/input/global-wheat-detection/train.csv")
#test = pd.read_csv("/kaggle/input/global-wheat-detection/test.csv")


# In[ ]:


#convert bbox into XMin XMax YMin YMax
#train["bbox"][0][1:-1].split(', ')
box_dict = {'XMin':[], 'XMax':[], 'YMin':[], 'YMax':[]}
def bbox(x):
    #x = X[1:-1].split(', ')
    #min value
    #print(x)
    box_dict['XMin'].append(int(x[0]))
    box_dict['YMin'].append(int(x[1]))
    
    #Max value
    box_dict['XMax'].append(int(x[0])+int(x[2]))
    box_dict['YMax'].append(int(x[1])+int(x[3]))
    

b = train["bbox"].apply(lambda X: bbox(list(map(float,X[1:-1].split(', ')))))
box = pd.DataFrame(box_dict)


# # Combine two dataframe image_id with annotation

# In[ ]:


train = pd.concat([train,box],axis=1)
train.drop("bbox",axis=1,inplace=True)


# In[ ]:


dir_train = os.listdir("/kaggle/input/global-wheat-detection/train/")
dir_test = os.listdir("/kaggle/input/global-wheat-detection/test/")


# In[ ]:


len(dir_test)


# In[ ]:


#Creating full path for each image ID
def creatingPath(x):
    return ("/kaggle/input/global-wheat-detection/train/"+x+".jpg")

train["path"] = train["image_id"].apply(creatingPath)


# In[ ]:


#Plot some of the images
def plot_bbox(img_id):
  img_url = train.loc[train["image_id"]==img_id]['path'].values[0]
  img = io.imread(img_url)
  height, width, channel = img.shape
  print(f"Image: {img.shape}")
  bboxs = train[train['image_id']==img_id]
  for index, row in bboxs.iterrows():
      xmin = row['XMin']
      xmax = row['XMax']
      ymin = row['YMin']
      ymax = row['YMax']
      #xmin = int(xmin*width)
      #xmax = int(xmax*width)
      #ymin = int(ymin*height)
      #ymax = int(ymax*height)
      label_name = row['source']
      class_series = train[train["source"]==label_name]
      class_name = class_series["source"].values[0]
      print(f"Coordinates: {xmin,ymin}, {xmax,ymax}")
      cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (255,0,0), 5)
      font = cv2.FONT_HERSHEY_SIMPLEX
      cv2.putText(img, class_name, (xmin,ymin-10), font, 1, (0,255,0), 5)
  plt.figure(figsize=(15,10))
  plt.title('Image with Bounding Box')
  plt.imshow(img)
  plt.axis("off")
  plt.show()


# In[ ]:


least_objects_img_ids = train["image_id"].value_counts().tail(50).index.values
for img_id in random.sample(list(least_objects_img_ids), 5):
  plot_bbox(img_id)


# In[ ]:


pd.read_csv("/kaggle/input/global-wheat-detection/train.csv").to_csv("/kaggle/input/global-wheat-detection/submission.csv")


# In[ ]:




