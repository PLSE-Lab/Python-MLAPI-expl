#!/usr/bin/env python
# coding: utf-8

# # <html> <h1> <font color="#5d5c2c"> Prostate cANcer graDe Assessment (PANDA) Challenge: Starter EDA </font> </h1>
# <hr/>
#     
# The aim of the PANDA challenge is to classify the severity of prostate cancer from images of prostate tissue samples. 
# 
# **How do you estimate the severity of prostate cancer (PCa)?** <br/>
# The diagnosis of prostate cancer is based on the grading of biopsies of prostate tissue and these are the images provided to us in the competition. These biopsies are scored by pathologists according to the Gleason grading system which classifies cancerous tissue into Gleason patterns (3, 4, or 5) based on the growth patterns of the tumor. The Gleason score is then converted to an ISUP grade on a 1-5 scale. You can take a look at the image below for an illustration of the process.
# 
# ![](https://storage.googleapis.com/kaggle-media/competitions/PANDA/Screen%20Shot%202020-04-08%20at%202.03.53%20PM.png)
# 
# Now that we have some background information, let's take a look at the data!

# In[ ]:


import numpy as np 
import pandas as pd 

import skimage.io
import openslide
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tifffile as tiff
from skimage.io import MultiImage
import gc

import PIL
from IPython.display import Image, display, HTML

plt.style.use('fivethirtyeight')

# Location of the training images
data_dir = '/kaggle/input/prostate-cancer-grade-assessment/train_images'
mask_dir = '/kaggle/input/prostate-cancer-grade-assessment/train_label_masks'

train_images = os.listdir(data_dir)
train_masks = os.listdir(mask_dir)


# In[ ]:


train = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv')
test = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/test.csv')
sample_submission = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/sample_submission.csv')

train.head()


# In[ ]:


print('Number of images: ', len(train_images))
print('Number of masks: ', len(train_masks))
print('Shape of the training data: ', train.shape)
print('Shape of the test data: ', test.shape)


# Before we go further, let's understand some of these attributes. Details taken from [this link](https://www.kaggle.com/c/prostate-cancer-grade-assessment/data). <br/>
# * image_id : ID for the training image
# * data_provider: The name of the institution that provided the data.
# * isup_grade: The target variable. The severity of the cancer on a 0-5 scale.
# * gleason_score: Train only. An alternate cancer severity rating system with more levels than the ISUP scale. 
# <p><br/></p>
# And these are the images that are provided:
# * train/test images: The images of prostate tissue samples
# * train_label_masks: Segmentation masks showing which parts of the image led to the ISUP grade.
# 
# **Note: The segmentation masks are missing for some images as there are 10616 train images and only 10516 masks.
# **
# 

# # <html> <h1> <font color="#5d5c2c"> EDA and Visualizations </font> </h1>
# <hr/>
#     
# First, we'll take a look at a few images in detail.

# In[ ]:


sns.set_style('whitegrid')
plt.rc('axes',edgecolor='#dddddd')
fig = plt.figure()
fig.set_size_inches(15, 25)

for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.tick_params(labelsize = 10)
    
    slide = MultiImage(os.path.join(data_dir, train_images[i]))

    ax = plt.gca()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5)
    ax.grid(False)      
    ax.set_title('Image: '+train_images[i], fontsize=10)
    ax.imshow(slide[-1])   


# <h2> Image Dimensions</h2>
# The data comes from two dato providers so now we'll take a look at the distribution of images between the two data providers. 

# In[ ]:


sns.set_style('whitegrid')
fig,ax=plt.subplots(1,2,figsize=(15,5))
train['data_provider'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0], colors=['#440154FF','#20A387FF'], pctdistance=1.1,labeldistance=1.2)
ax[0].set_ylabel('')
sns.countplot('data_provider',data=train, ax=ax[1], palette=['#440154FF','#20A387FF'])
fig.suptitle('Distribution of Images between the Data Providers', fontsize = 14)
plt.show()


# <h2> Image Dimensions</h2>
# From our quick look at the images we can tell that there is quite a bit of variation in the image dimensions. We'll plot the image dimensions below to take a closer look.

# In[ ]:


from tqdm import tqdm

image_dimensions = []

for i,row in tqdm(train.iterrows()):
    slide = openslide.OpenSlide(os.path.join(data_dir, train.image_id.iloc[i]+'.tiff'))
    image_dimensions.append(slide.dimensions)
    slide.close()
    
width = [dimensions[0] for dimensions in image_dimensions] 
height = [dimensions[1] for dimensions in image_dimensions] 

train['width'] = width
train['height'] = height


# In[ ]:


sns.set_style('whitegrid')
fig = plt.figure(figsize=(18,8))
ax = sns.scatterplot(x='width', y='height', data=train, color='#440154FF', alpha=0.5)
ax.tick_params(labelsize=10)
plt.title('Dimensions of Images')
plt.show()


# There is quite a bit of variation in the image dimensions - and some of them are very large!

# <h3> Image Dimensions by Data Provider - Karolinska Vs Radboud</h3>
# 

# In[ ]:


sns.set_style('whitegrid')
fig = plt.figure(figsize=(18,8))
ax = sns.scatterplot(x='width', y='height', data=train, hue='data_provider', palette=['#440154FF','#20A387FF'], alpha=0.70)
ax.tick_params(labelsize=10)
plt.title('Dimensions of Images')
plt.show()


# In[ ]:


fig, ax = plt.subplots(1, 2)
fig.set_size_inches(15, 5)

sns.stripplot(train['width'],train['data_provider'],ax=ax[0], palette=['#440154FF','#20A387FF'],jitter=True)
sns.stripplot(train['height'],train['data_provider'],ax=ax[1], palette=['#440154FF','#20A387FF'],jitter=True)

ax[0].tick_params(labelsize=10)
ax[1].tick_params(labelsize=10)
ax[0].tick_params(labelrotation=90)
ax[1].tick_params(labelrotation=90)
plt.show()


# The images from Karolinska are larger in general and it also has a few that are extremely large.  

# <h2> Isup Grades and Gleason Scores </h2>
# Now, lets take a look at the data that indicates the severity of cancer (or absence). We have the following:
# * isup_grade, which is the target variable, and measures the severity of the cancer on a 0-5 scale 
# * gleason_score which is an alternate cancer severity rating system with more levels than the ISUP scale

# In[ ]:


fig, ax = plt.subplots(1, 2)
fig.set_size_inches(15, 6)

sns.countplot(x='isup_grade', data = train, ax=ax[0], palette='viridis',order = train['isup_grade'].value_counts().index)
sns.countplot(x='gleason_score', data = train, ax=ax[1], palette='viridis', order = train['gleason_score'].value_counts().index)

ax[0].set_title('ISUP Grade (target variable)',y=1.0, fontsize = 14)
ax[1].set_title('Gleason Score', y=1.0, fontsize = 14)

for axis in ['top','bottom','left','right']:
    ax[0].spines[axis].set_linewidth(0.7)
    ax[1].spines[axis].set_linewidth(0.7)
    
ax[0].tick_params(labelsize=10)
ax[1].tick_params(labelsize=10)

plt.tight_layout()
plt.subplots_adjust(hspace = 0.2)
plt.show()


# <h2> Mapping of the Gleason Scores to ISUP Grades</h2>
# 
# The Gleason Scores can be mapped to the  ISUP Grades and the mapping is given in the image below. 
# 
# ![](https://storage.googleapis.com/kaggle-media/competitions/PANDA/Screen%20Shot%202020-04-08%20at%202.03.53%20PM.png)
# 

# **Further, in this challenge, biopsies that do not indicate cancer given an ISUP grade of 0.
# **
# <br/>
# Let's check if our data matches all this.

# In[ ]:


train_grouped = train.groupby('isup_grade')['gleason_score'].unique().to_frame().reset_index()
train_grouped.style.highlight_min(color='yellow')


# There seem to be two inconsistencies:
# * We have an extra 'negative' value that corresponds to isup_grade 0
# * The isup_grade 2 is only mapped to gleason_score '3+4' but in our data it has '4+3' as well. 

# First we'll take a look at the 'negative' values in the gleason_score. 
# 
# We have plotted the gleason_score by data_provider below and we can see that radboud has no '0+0' values while karolinska has no 'negative' values. 'negative' is just how radbound represents '0+0' or the absence of cancer so it might make sense to replace 'negative' with '0+0' - will decide later. 

# In[ ]:


fig = plt.figure(figsize=(10, 5))
sns.set_style('whitegrid')

#sns.countplot(x='isup_grade', hue='data_provider', data = train, ax=ax[0], palette='viridis_r')
sns.countplot(x='gleason_score', hue='data_provider', data = train, palette=['#440154FF','#20A387FF'])
plt.title('Gleason Score by Data Provider')
plt.tick_params(labelsize=10)

plt.tight_layout()
plt.show()


# Next, we'll take a look at the rows that have incorrectly mapped gleason_score and isup_grade.

# In[ ]:


train[(train.isup_grade == 2) & (train.gleason_score == '4+3')].reset_index()


# There is just one image and it seems like an error so let's drop it and look at our data (grouped by isup_grade).

# In[ ]:


train.reset_index(inplace=True)
train = train[train.image_id !='b0a92a74cb53899311acc30b7405e101']


# In[ ]:


train_grouped = train.groupby('isup_grade')['gleason_score'].unique().to_frame().reset_index()
train_grouped.style.highlight_min(color='yellow')


# <h2> Images with their masks</h2>
# Next, lets take a quick look at the images and the masks.

# In[ ]:


# Plot the images and their corresponding masks
plt.rc('axes',edgecolor='#dddddd')

for i in range(4):
    slide = MultiImage(os.path.join(data_dir, train_images[i]))
    maskfile = MultiImage(os.path.join(mask_dir, train_images[i][:-5]+'_mask.tiff'))
    mask_level_2 = maskfile[-1][:,:,0]    

    fig, (ax1, ax2) = plt.subplots(1, 2)    
    fig.set_size_inches(15, 5)

    ax1.grid(False)
    ax2.grid(False)
    ax1.set_title('Image: '+train_images[i], fontsize=10)
    ax2.set_title('Image: '+train_images[i]+ ' with Mask', fontsize=10)

    ax1.imshow(slide[-1])   
    ax2.imshow(mask_level_2)    
    plt.show()


# In[ ]:


del slide # Free up memory
del maskfile
gc.collect()


# The masks show which part of the tissue image led to the ISUP grade and the value depends on the data provider. The values are taken from [here](https://www.kaggle.com/c/prostate-cancer-grade-assessment/data) and given below: <br/>
# <p><br/></p>
# Radboud: Prostate glands are individually labelled. Valid values are:
# * 0: background (non tissue) or unknown
# * 1: stroma (connective tissue, non-epithelium tissue)
# * 2: healthy (benign) epithelium
# * 3: cancerous epithelium (Gleason 3)
# * 4: cancerous epithelium (Gleason 4)
# * 5: cancerous epithelium (Gleason 5)
# 
# Karolinska: Regions are labelled. Valid values are:
# * 1: background (non tissue) or unknown
# * 2: benign tissue (stroma and epithelium combined)
# * 3: cancerous tissue (stroma and epithelium combined)

# 
# This notebook is still a work in progress. Please upvote if you like it!
