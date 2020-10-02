#!/usr/bin/env python
# coding: utf-8

# 
# This kernel is an attempt at understanding what we have. This is the first of the series of kernels. 
# 
# 1. [Exploring csv files](#Exploring-CSV-files)
#     - [train.csv](#train.csv)
#     - [test.csv](#test.csv)
#     - [class_maps.csv](#class_maps.csv)
#     - [sample_submission.csv](#sample_submission.csv)
# 2. [Exploring parquet files](#Exploring-parquet-files)
# 
# 

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc # garbage collector

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


project_dir = '/kaggle/input/bengaliai-cv19/'


# In[ ]:


import glob
csv_files = [file for file in glob.glob(project_dir+"*.csv")]
train_parquet_files =  [file for file in glob.glob(project_dir+"train*.parquet")]
test_parquet_files =  [file for file in glob.glob(project_dir+"test*.parquet")]


# ---
# 
# ## Exploring CSV files
# 

# In[ ]:


csv_files


# In[ ]:


sample_submission = '/kaggle/input/bengaliai-cv19/sample_submission.csv'
class_maps = '/kaggle/input/bengaliai-cv19/class_map.csv'
test = '/kaggle/input/bengaliai-cv19/test.csv'
train = '/kaggle/input/bengaliai-cv19/train.csv'

# For some strange reason, when I commit csv_files order is changing

# sample_submission = csv_files[0]
# class_maps = csv_files[1]
# test = csv_files[2]
# train = csv_files[3]


# In[ ]:


def csv_overview(csv_file, name='', head=3, tail=3, columns=False, describe=False, info=True):
    print('file :', csv_file)
    df = pd.read_csv(csv_file)
    print('{} Shape : '.format(name),df.shape)
    print('-'*36)
    if columns:
        print('{} Columns : '.format(name),df.columns)
        print('-'*36)
    if describe:
        print('{} Distribution :\n'.format(name),df.describe().T)
        print('-'*36)
    if info:
        print('{} Summary :\n'.format(name))
#         print(df.info())
        df.info()
        print('-'*36)
    print('{} Unique values :\n'.format(name),df.nunique())
    print('-'*36)
    print('Sample data')
    print('-'*12)
    print('head')
    print(df.head(head))
    print('-'*12)
    print('tail')
    print(df.tail(tail))


# ### train.csv

# In[ ]:


csv_overview(train, 'train_df', columns=True)


# #### Overview :
# There are total of 200840 (~200K images) and each image has 
# - grapheme_root
# - vowel_diacritic
# - consonant_diacritic
# - and grapheme (which is combination of above three), is the bengali character in image
# 
# > Task at hand is to identify the compoments which makeup the grapheme
# 
# #### Summary :
# 
# From train.csv, we can observe
# 
# -  168 unique grapheme_root 
# -  11 unique vowel_diacritic
# -  7 unique consonant_diacritic
# -  1295 unique graphemes 
# 
# Images are identified using image_id col(Train_image-no) which are likely to be foreign keys in paraquet files
# 
# > There are doesn't seem to be any null values
# 
# 
# ---

# ### test.csv

# In[ ]:


csv_overview(test, 'test_df', head=5, tail=5, columns=True)


# Each images seems to have 3 dedicated rows to it each for different type of component.
# 
# Given test.csv only have test data for 12 images compared to train.csv with over ~200K images. These shall be used only for sample submission.
# 
# > Goal would be to predict the component given its corresponding row_id and/or image_id
# 
# ---
# 

# ### class_maps.csv

# In[ ]:


csv_overview(class_maps, 'Class Maps', columns=True)


# Each component type (represented by a unique label) is mapped to the font/visualization
# 
# we have a total of 168+11+7 -> 186 components
# 
# ---

# ### sample_submission.csv

# In[ ]:


csv_overview(sample_submission, 'Sample Submissions', head=5, columns=True)


# Submission should have only 2 columns, one for image and component identification and the other for its label
# 
# ---

# ## Exploring parquet files

# In[ ]:


def explore_parquet(file, name='', head=3, tail=3, columns=False, describe=False, unique=False, info=True):
    print('file : {}'.format(file))
    df = pd.read_parquet(file)
    print('{} Shape : '.format(name),df.shape)
    print('-'*36)
    if columns:
        print('{} Columns : '.format(name),df.columns)
        print('-'*36)
    if describe:
        print('{} Distribution :\n'.format(name),df.describe().T)
        print('-'*36)
    if info:
        print('{} Summary :\n'.format(name))
#         print(df.info())
        df.info()
        print('-'*36)
    if unique:
        print('{} Unique values :\n'.format(name),df.nunique())
        print('-'*36)
    print('Sample data')
    print('-'*12)
    print('head')
    print(df.head(head))
    print('-'*12)
    print('tail')
    print(df.tail(tail))


# In[ ]:


def visualize_parquet(file, shape=(137, 236), cmap=None):
    # shape - (height, width)
    df = pd.read_parquet(file)
    df1 = df.head(25)
    labels, images = df1.iloc[:, 0], df1.iloc[:, 1:].values.reshape(-1, *shape) 
    
    f, ax = plt.subplots(5, 4, figsize=(20, 20))
    ax = ax.flatten()
    f.suptitle(file) #super title
    
    for i in range(20):
        ax[i].set_title(labels[i])
        ax[i].imshow(images[i], cmap=cmap)


# ### Train parquet files

# In[ ]:


train_parquet_files


# In[ ]:


for file in train_parquet_files:
    explore_parquet(file)
    print('==='*18)


# We can see that each parquet file has ~50k images of the total 200K train images.
# 
# Out of 32333 columns
# 
# - First column is the Foriegn key for the image from train.csv
# - The next 32332 columns are pixels of the flattend image of size 137x236

# In[ ]:


for file in train_parquet_files:
    visualize_parquet(file, cmap='Blues')


# In[ ]:





# In[ ]:





# ### Test parquet files

# In[ ]:


test_parquet_files


# In[ ]:


for file in test_parquet_files:
    explore_parquet(file)
    print('==='*18)
        


# Each test parquet file has 3 images - flattened along with the image_id**

# In[ ]:


shape=(137, 236)
for file in test_parquet_files:
    print('file', file)
    df = pd.read_parquet(file)
    labels, images = df.iloc[:, 0], df.iloc[:, 1:].values.reshape(-1, *shape) 

    f, ax = plt.subplots(1, 3, figsize=(10, 10))
    ax = ax.flatten()
#     f.suptitle(file) #super title

    for i in range(3):
        ax[i].set_title(labels[i])
        ax[i].imshow(images[i], cmap='Blues')


# In[ ]:




