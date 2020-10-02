#!/usr/bin/env python
# coding: utf-8

# # Artificial Intelligence
# ## X-ray Pneumonia Classification
# 10/22/2018  
# * Bethelhem Tesfaw
# * John Wolsky
# * Liu Huimin
# * Naomi Sprague
# * Garth Mortensen

# ### Problem
# #### Part 1: Classify Images
# Using Convolutional Neural Networks (CNN) and other machine learning methods, classify chest x-ray images as:
# 
# 1. Healthy 
# 2. Pneumonia viral 
# 3. Pneumonia bacterial
# 
# #### Part 2: Visualize what CNNs Learn
# After classifying these images, visualize and interpret the learned classified representations. Much research has been done to determine why a neural network produces certain results, given predictors. That is, explore how the CNN classified the images. Three ways to accomplish this are:
# 
# 1. Visualize intermediate convnet outputs (intermediate activations) - *Show how succesive layers tranform output, and see the first idea of individual convnet filters.*
# 2. Visualize convnets filters - *See precisely what visual pattern or concept each filter is receptive to.*
# 3. Visualize heatmaps of class activation in an image - *Identify which parts of an image were identified as belonging to a certain class.*
# 
# **Deep Learning in Python (Ch. 5, pg 160)**, can serve as a guide for the second half of this project. On Monday, October 8, 2018 at 5:48:26 PM plus 1h17m, Professor Lai also discussed how to identify driving factors in prediction. Simplified, we can take the output of a hidden layer, and build a machine learning model to predict. The weights of this model indicates the relative importance of each factor.
# 
# Through this work, we will not have only classified the x-ray images, but also opened the black box to understand its operatations and determination of important predictors.
#   
# ### Scope
# Initially, we will build a CNN to classify the x-ray images into the three classes. Various architectures will be used before settling on a CNN, perhaps including AlexNet, VGG-16 and others. The CNN's performance will then be measured using accuracy and other metrics, such as ROC.
#  
# The CNN's x-hat hidden layer outputs may then be used as inputs to build several machine learning models. To determine the effecacy these models, their performance would be evaluated. As a possible final step, we may combine them into an ensemble model, with the same end goal of producing a three-way classification.
#  
# ### Data
# #### Data Source
# The data archive is housed on kaggle, here:
# https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
# 
# 
# #### Data Structure
# The dataset is a .zip archive containing the three pre-split folders of training, validation and test. Housed within them are a total of 5,863 images. The data is pre-split into those folders, with subfolders for normal and pneumonia. 
#  
# The archive is 1.21Gb zipped, and 1.17Gb uncompressed. The directory structure is shown below.
# 
# * /chest_xray/  
#   * test/
#     * NORMAL/
#     * PNEUMONIA/
#   * train/
#     * NORMAL/
#     * PNEUMONIA/
#   * val/
#     * NORMAL/
#     * PNEUMONIA/
# 
# These folders will provide class labels.
# 
# This folder structure does not follow a 70-30 split, but rather a 89-1-10. It is an odd split, which we may later erase by merging all folders and splitting according to our desires. 
# 
# Our dataset is significantly imbalanced. there are nearly *3x the amount of pneumonia labels than normal labels*. However, in training data, we do have two classes of pneumonia, so the imbalance may be less severe than it appears at first glance. This ratio doesn't hold equal for test data. Some time should be reserved to read further into working with unbalanced data.
# 
# 
# |**Dir**    |                               | **Files**     | **% Total**|
# |-------|-----------|-------:|---------:|
# | test  |                                |               | 10.6  |
# |         | NORMAL              | 234        | 4.0   |
# |         | PNEUMONIA       | 390       | 6.7    |
# | train |                              |                | 89.0 |
# |         | NORMAL             | 1,342      | 22.9 |
# |         | PNEUMONIA      | 3,876     | 66.1  |
# | val   |                               |                | 0.3   |
# |         | NORMAL             | 9            | 0.2   |
# |         | PNEUMONIA      | 9            | 0.2   |
# | **Total** |                             | **5,860**    | **100**       |
# .
# 
# The data is comprised entirely of images whose properties are:
# * Of varying pixel height x width. **POTENTIAL TROUBLE** / [*POTENTIAL SOLUTION*](https://www.kaggle.com/dansbecker/programming-in-tensorflow-and-keras)
# * 8-bit depth
# * grayscale  
# 
# #### Potential Challenges
# 
# To load this data, we might use this [kernel](https://www.kaggle.com/jgrandinetti/classification-using-convnet-keras) as a guide. ImageDataGenerator uses flow_from_directory, which could be a perfect solution for this folder structure.
# 
# It remains to be seen whether these large format images will pose processing (CPU/GPU) and memory (RAM) challenges, but downsampling might be helpful. Pooling early on could decrease processing time substantially.
# 
# ### Tools
# We will use the following tools:
# 1. Kaggle
# 2. MATLAB
# 3. Python  
#     I. Tensorflow  
#     II. Keras  
# 4. Google Drive
# 5. Onedrive

# ## Initial Data Review

# In[ ]:


# Basic math libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Import library
import os
print(os.listdir("../input"))
# Take note that the .7z directory name is automatically converted to lowercase. 

import os.path
# Ensure we're reading the directory correctly.
os.path.exists('../input/chest_xray/chest_xray')


# First, locate the main directory and display its content.

# In[ ]:


directory = os.listdir('../input/chest_xray/chest_xray')
print("Parent directory includes these folders:", directory)


# The folders are split to include train, testing, as well as validation. Create variables pointing to these subfolders

# In[ ]:


train_folder = '../input/chest_xray/chest_xray/train/'
val_folder   = '../input/chest_xray/chest_xray/val/'
test_folder  = '../input/chest_xray/chest_xray/test/'


# Define the two training class folders of Normal and Pneumonia.

# ### Load Data
# #### Define Training Data

# In[ ]:


# train 
os.listdir(train_folder)
train_n = train_folder + 'NORMAL/'
train_p = train_folder + 'PNEUMONIA/'


# We randomly select one image from the healthy hat.

# In[ ]:


# Normal pic 
print("Total images this directory are:", len(os.listdir(train_n)))

rand_norm = np.random.randint(0, len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
norm_pic_address = train_n + norm_pic

print('normal picture title: ', norm_pic)


# And we randomly select one image from the pneumonia hat.

# In[ ]:


# Pneumonia
print("Total images this directory are:", len(os.listdir(train_p)))

rand_p = np.random.randint(0, len(os.listdir(train_p)))
sic_pic =  os.listdir(train_p)[rand_p]
sic_pic_address = train_p + sic_pic

print('pneumonia picture title:', sic_pic)


# Using Python Image Library (PIL), we can open the images.

# In[ ]:


# Load the images
# Image.open is from Import
from PIL import Image

norm_load = Image.open(norm_pic_address)
sic_load  = Image.open(sic_pic_address)


# And plot them.

# In[ ]:


# plot images
f = plt.figure(figsize = (10, 10))

# add_subplot(nrows, ncols, index, **kwargs)
a1 = f.add_subplot(1, 2, 1)
img_plot = plt.imshow(norm_load)
a1.set_title('Normal')

a2 = f.add_subplot(1, 2, 2)
img_plot = plt.imshow(sic_load)
a2.set_title('Patient-Zero')


# Is there anything evident to you in the images above? I see nothing at all.

# #### Define Validation Data

# In[ ]:


# validation 
os.listdir(val_folder)
val_n = val_folder + 'NORMAL/'
val_p = val_folder + 'PNEUMONIA/'


# Select one validation data normal image.

# In[ ]:


# Normal pic 
print("Total images this directory are:", len(os.listdir(val_n)))

rand_norm = np.random.randint(0, len(os.listdir(val_n)))
norm_pic = os.listdir(val_n)[rand_norm]
norm_pic_address = val_n + norm_pic

print('normal picture title: ', norm_pic)


# And select a Pneumonia image.

# In[ ]:


# Pneumonia
print("Total images this directory are:", len(os.listdir(val_p)))

rand_p = np.random.randint(0, len(os.listdir(val_p)))
sic_pic =  os.listdir(val_p)[rand_p]
sic_pic_address = val_p + sic_pic

print('pneumonia picture title:', sic_pic)


# #### Define Test Data

# In[ ]:


# test
os.listdir(test_folder)
test_n = test_folder + 'NORMAL/'
test_p = test_folder + 'PNEUMONIA/'


# Select one test data image.

# In[ ]:


# Normal pic 
print("Total images this directory are:", len(os.listdir(test_n)))

rand_norm = np.random.randint(0, len(os.listdir(test_n)))
norm_pic = os.listdir(test_n)[rand_norm]
norm_pic_address = test_n + norm_pic

print('normal picture title: ', norm_pic)


# And select a pneumonia image.

# In[ ]:


# Pneumonia
print("Total images this directory are:", len(os.listdir(test_p)))

rand_p = np.random.randint(0, len(os.listdir(test_p)))
sic_pic =  os.listdir(test_p)[rand_p]
sic_pic_address = test_p + sic_pic

print('pneumonia picture title:', sic_pic)


# ### Consider Imbalanced Dataset
# **This dataset is highly imbalanced, which is problematic because:** Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Sollicitudin tempor id eu nisl nunc mi ipsum. Pellentesque eu tincidunt tortor aliquam nulla facilisi cras fermentum odio. Interdum consectetur libero id faucibus nisl tincidunt eget. Mattis aliquam faucibus purus in. Dignissim cras tincidunt lobortis feugiat vivamus at augue eget. Purus faucibus ornare suspendisse sed nisi lacus sed viverra tellus. Enim tortor at auctor urna nunc id cursus metus aliquam.

# In[ ]:


print("Total train normal images:", len(os.listdir(train_n)))
print("Total validation normal images:", len(os.listdir(val_n)))
print("Total test normal images:", len(os.listdir(test_n)))


# In[ ]:


print("Total train pneumonia images:", len(os.listdir(train_p)))
print("Total validation pneumonia images:", len(os.listdir(val_p)))
print("Total test pneumonia images:", len(os.listdir(test_p)))


# Group all these counts together for their totals.

# In[ ]:


train_n_count = len(os.listdir(train_n))
val_n_count   = len(os.listdir(val_n))
test_n_count  = len(os.listdir(test_n))

train_p_count = len(os.listdir(train_p))
val_p_count   = len(os.listdir(val_p))
test_p_count  = len(os.listdir(test_p))

normal_count = [train_n_count, val_n_count, test_n_count]
sic_count = [train_p_count, val_p_count, test_p_count]

print(normal_count)
print(sic_count)


# Add them up to compare simply the total normal vs pneumonia counts.

# In[ ]:


normal_count_total = train_n_count + val_n_count + test_n_count
sic_count_total    = train_p_count + val_p_count + test_p_count

print(normal_count_total)
print(sic_count_total)


# Plot the results for interpretation.

# In[ ]:


# data to plot
n_groups = 1

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
 
rects1 = plt.bar(index, normal_count_total, bar_width,
                 alpha = opacity,
                 color = 'b',
                 label = 'normal')
 
rects2 = plt.bar(index + bar_width, sic_count_total, bar_width,
                 alpha = opacity,
                 color = 'r',
                 label = 'pneumonia')
 
plt.xlabel('Dataset')
plt.ylabel('Image Count')
plt.title('Total Normal-Pneumonia Imbalance')
plt.xticks(index + bar_width, ('', ''))
plt.legend()
 
plt.tight_layout()
plt.show()


# Display percentages of total.

# In[ ]:


total_images = normal_count_total + sic_count_total

normal_count_pct = normal_count_total / total_images
sic_count_pct = sic_count_total / total_images

print("Percentage normal: {0} %\n".format(normal_count_pct))
print("Percentage pneumonia: {0} %\n".format(sic_count_pct))


# There is a severe imbalance between the labels. We can drill down and visualize these distribution measures using a [bar chart](https://pythonspot.com/matplotlib-bar-chart/).

# In[ ]:


# data to plot
n_groups = 3
    
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
 
rects1 = plt.bar(index, normal_count, bar_width,
                 alpha = opacity,
                 color = 'b',
                 label = 'normal')
 
rects2 = plt.bar(index + bar_width, sic_count, bar_width,
                 alpha = opacity,
                 color = 'r',
                 label = 'pneumonia')
 
plt.xlabel('Dataset')
plt.ylabel('Image Count')
plt.title('Train-Validation-Test Imbalance')
plt.xticks(index + bar_width, ('Train', 'Validation', 'Test'))
plt.legend()
 
plt.tight_layout()
plt.show()


# Validation is exceptionally small in count, whereas overall, training pneumonia takes the majority of all subsets.

# ## Data Split
# The data should be split to validate the efficacy of the models. We might use either cross validation, or a 70-30 split.
# 
# **Before splitting the data,  we will need to obtain the y-labels to go along with X.**

# ### Obtain Y-labels

# In[ ]:


# from sklearn.model_selection import train_test_split

#Splitting 
# X_train, X_test, y_train, y_test = train_test_split(x_files, y_train, test_size = 0.3, random_state = 42)


# ## Project Proposal End

# This concludes our initial examination of our data. Our next step is to build a CNN, tweak its architecture, and explore other available models for import. The best classification accuracy will be chosen from the models.
# 
# After completing the first half of this project, we will then turn to interpretting and visualizing the model, using various methods.
# 
# ### Thank you

# In[ ]:




