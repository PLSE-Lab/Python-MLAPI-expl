#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This kernel is just a quick look at the training dataset image sizes, and a look at some of the images at the lowest and highest diagnosis levels. To see if there is something easily visible to understand what the doctor might be looking at in a classification.
# 
# There is also a [previous competition](https://www.kaggle.com/c/diabetic-retinopathy-detection) on the same topic, with the exact same training labels. It seems to have a much larger training dataset. This set was mentioned multiple times in the [external data thread](). I had trouble adding that competition as a data source (error about loading the data). So I downloaded the data and set it up as a [separate dataset](https://www.kaggle.com/donkeys/retinopathy-train-2015). Had to downscale it quite a bit to max 896x896 pixel sizes, to fit it into the 20GB dataset size limit. But it seems potentially useful.
# 
# I am not quite sure how to check exact date of some old competition here on Kaggle, so I just picket the number of years it displays in the past, and went with 2015. So I will call the older set the *2015* set here. Or the *past* set vs the actual current set for the *present* time.
# 

# In[ ]:


import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import math
import PIL
from PIL import ImageOps
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Activation, Dropout, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, applications
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras import backend as K 
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import keras

from tqdm.auto import tqdm
tqdm.pandas()


# In[ ]:


get_ipython().system('ls -l ../input/')


# ## Number of files in train vs test vs the 2015 training set

# In[ ]:


get_ipython().system('ls -l ../input/aptos2019-blindness-detection/train_images | wc -l')


# In[ ]:


get_ipython().system('ls -l ../input/aptos2019-blindness-detection/test_images | wc -l')


# In[ ]:


get_ipython().system('ls -l ../input/retinopathy-train-2015/rescaled_train_896/rescaled_train_896 | wc -l')


# ## Basic metadata

# In[ ]:


train_path_2015 = "../input/retinopathy-train-2015/rescaled_train_896/rescaled_train_896/"
train_path = "../input/aptos2019-blindness-detection/train_images/"
test_path = "../input/aptos2019-blindness-detection/test_imges/"


# In[ ]:


df_train = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
df_train.head()


# In[ ]:


df_test = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
df_test.head()


# In[ ]:


df_train_2015 = pd.read_csv("../input/retinopathy-train-2015/rescaled_train_896/trainLabels.csv")
df_train_2015.head()


# First 10 un-ordered files in past and present training sets to see the filenames match the csv columns ("id_code" and "image"):

# In[ ]:


get_ipython().system('ls -lU ../input/aptos2019-blindness-detection/train_images/ | head -10')


# In[ ]:


get_ipython().system('ls -lU ../input/retinopathy-train-2015/rescaled_train_896/rescaled_train_896 | head -10')


# It's a match.

# ## Collect all metadata to single dataframe(s)

# In[ ]:


n_rows = df_train.shape[0]
n_rows


# In[ ]:


df_train["filename"] = df_train["id_code"]+".png"
df_train["path"] = [train_path]*n_rows
#the year is just to be able to easily separate the past and present datasets later
df_train["year"] = [2019]*n_rows
df_train.head()


# In[ ]:


n_rows_2015 = df_train_2015.shape[0]
n_rows_2015


# In[ ]:


df_train_2015["filename"] = df_train_2015["image"]+".png"
df_train_2015["path"] = [train_path_2015]*n_rows_2015
df_train_2015["year"] = [2015]*n_rows_2015
df_train_2015.head()


# In[ ]:


df_train_2015.columns = ["id_code", "diagnosis", "filename", "path", "year"]
df_train_2015.head()


# In[ ]:


df_train_all = pd.concat([df_train,df_train_2015], axis=0, sort=False).reset_index()
df_train_all.head()


# In[ ]:


df_train_all.tail()


# In[ ]:


#replacing df_train with the full set to calculate features and do visualizations all at once, keeping the original (present) just in case
df_train_orig = df_train
df_train = df_train_all


# ## Calculate Aspect Ratios etc.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'img_sizes = []\nwidths = []\nheights = []\naspect_ratios = []\n\nfor index, row in tqdm(df_train.iterrows(), total=df_train.shape[0]):\n    filename = row["filename"]\n    path = row["path"]\n    img_path = os.path.join(path, filename)\n    with open(img_path, \'rb\') as f:\n        img = PIL.Image.open(f)\n        img_size = img.size\n        img_sizes.append(img_size)\n        widths.append(img_size[0])\n        heights.append(img_size[1])\n        aspect_ratios.append(img_size[0]/img_size[1])\n\ndf_train["width"] = widths\ndf_train["height"] = heights\ndf_train["aspect_ratio"] = aspect_ratios\ndf_train["size"] = img_sizes')


# In[ ]:


df_train.head()


# ## Aspect Ratios
# 
# See that there are no images that are hugely different in size to others:

# In[ ]:


df_sorted = df_train.sort_values(by="aspect_ratio")


# In[ ]:


df_sorted.head()


# ### Past

# In[ ]:


df_sorted[df_sorted["year"] == 2015].head()


# ### Present

# In[ ]:


df_sorted[df_sorted["year"] == 2019].head()


# The aspect ratios in the past and present seem very close to each other.

# In[ ]:


df_sorted.tail()


# In[ ]:


df_sorted[df_sorted["year"] == 2015].tail()


# In[ ]:


df_sorted[df_sorted["year"] == 2019].tail()


# # Look at the Images / Eyes

# In[ ]:


#This just shows a single image in the notebook
def show_img(filename, path):
        img = PIL.Image.open(f"{path}/{filename}")
        npa = np.array(img)
        print(npa.shape)
        #https://stackoverflow.com/questions/35902302/discarding-alpha-channel-from-images-stored-as-numpy-arrays
#        npa3 = npa[ :, :, :3]
        print(filename)
        plt.imshow(npa)


# In[ ]:


import matplotlib

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)


# ## A Random Eye
# 
# Visualize the first image in past and present sets to see if they are at all alike:
# 

# ### Present

# In[ ]:


row = df_sorted[df_sorted["year"] == 2019].iloc[0]
show_img(row.filename, row.path)


# ### Past

# In[ ]:


row = df_sorted[df_sorted["year"] == 2015].iloc[0]
show_img(row.filename, row.path)


# ## 9-Eyes
# 
# Visualize 9 images from a set at a time, to learn a bit more about the set at once.

# In[ ]:


def plot_first_9(df_to_plot):
    plt.figure(figsize=[30,30])
    for x in range(9):
        path = df_to_plot.iloc[x].path
        filename = df_to_plot.iloc[x].filename
        img = PIL.Image.open(f"{path}/{filename}")
        print(filename)
        plt.subplot(3, 3, x+1)
        plt.imshow(img)
        title_str = filename+", diagnosis: "+str(df_to_plot.iloc[x].diagnosis)
        plt.title(title_str)


# ## Smallest Aspect Ratio
# 
# There seem to be no images with aspect ratio < 1, so plotting the smallest aspect ratios (practically the ratio is then 1) should show the most "square" images:

# In[ ]:


del df_sorted
df_sorted = df_train.sort_values(by="aspect_ratio", ascending=True)


# ### Present

# In[ ]:


plot_first_9(df_sorted[df_sorted["year"] == 2019])


# ### Past

# In[ ]:


plot_first_9(df_sorted[df_sorted["year"] == 2015])


# Generally, the past vs present images seem very similar. Some color differences, although some of the later pics will show both have these more "orange" and "greenish" ones as well. But a deeper investigation of how the color spaces are distributed in different sets could be interesting.

# ## Highest Aspect Ratios
# 
# This should be the ones least "square":

# In[ ]:


del df_sorted
df_sorted = df_train.sort_values(by="aspect_ratio", ascending=False)


# ### Present

# In[ ]:


plot_first_9(df_sorted[df_sorted["year"] == 2019])


# ### Past

# In[ ]:


plot_first_9(df_sorted[df_sorted["year"] == 2015])


# ## Diagnosis Values
# 
# A look at the highest vs lowest diagnosis values /levels given in the training set. Can we spot some differences? 
# 
# ### Highest / Most Severe Diagnosis:

# In[ ]:


del df_sorted
df_sorted = df_train.sort_values(by="diagnosis", ascending=False)
df_sorted.head()


# ### Present

# In[ ]:


plot_first_9(df_sorted[df_sorted["year"] == 2019])


# ### Past

# In[ ]:


plot_first_9(df_sorted[df_sorted["year"] == 2015])


# ### Lowest / Healthiest Diagnosis:

# In[ ]:


del df_sorted
df_sorted = df_train.sort_values(by="diagnosis", ascending=True)
df_sorted.head()


# ### Present

# In[ ]:


plot_first_9(df_sorted[df_sorted["year"] == 2019])


# ### Past

# In[ ]:


plot_first_9(df_sorted[df_sorted["year"] == 2015])


# I guess the healthier ones look more "clean".

# # Final Size Statistics
# 
# On average, are the files about the same size? Actually might make sense to look at the past and present sets separately since I had to downsize the past significantly. But the idea is there, and it does already show if there are some really small ones..

# In[ ]:


df_train.describe()


# Are the smallest files still valid files?

# In[ ]:


df_sorted = df_train.sort_values(by="width", ascending=True)

plot_first_9(df_sorted[df_sorted["year"] == 2019])


# In[ ]:



plot_first_9(df_sorted[df_sorted["year"] == 2015])


# # Conclusions
# 
# The images from both sets seem to be quite similar. Possibly some color differences and other minor differences?

# In[ ]:





# In[ ]:




