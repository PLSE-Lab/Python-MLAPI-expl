#!/usr/bin/env python
# coding: utf-8

# Hello, fellow Kagglers. We are going to look a very interesting problem today. This dataset consists of images of cats along with their facial landmarks. So, in short, this problem is to develop facial embeddings for cats. LOL!! Let's dive in
# ![](https://media.giphy.com/media/3o752eDHQgwl8zS3Oo/giphy.gif)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import cv2
import glob
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, imsave
from keras.preprocessing.image import load_img, array_to_img, img_to_array
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Input
from keras.optimizers import SGD, Adam, Adadelta, Adagrad
from keras import backend as K
from sklearn.model_selection import train_test_split
np.random.seed(111)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# The dataset is arranged in the following structure:
# ```
# --cats
#        --CAT_00
#                  - -.jpg
#                 - -.jpg.cat
#        --CAT_01
#                  - -*.jpg
#                  -- *.jpg.cat         
#         .............................
#         ..............................
#   ```      
# There are 7 subdirectories in total. Each of the subdirectory contains cat images and their corresponding facial key points annotations. The annotation file contains `.cat` extension in the end and contains landmarks in the following format:
#  `[num_keypoints x1 y1 x2 y2 x3 y3 ..........................x18 y18 '']`

# In[ ]:


# Define some Paths
input_path = Path('../input/cats')
cats = os.listdir(input_path)
print("Total number of sub-directories found: ", len(cats))

# Store the meta-data in a dataframe for convinience 
data = []
for folder in cats:
    new_dir = input_path / folder
    images = sorted(new_dir.glob('*.jpg'))
    annotations = sorted(new_dir.glob('*.cat'))
    n = len(images)
    for i in range(n):
        img = str(images[i])
        annotation = str(annotations[i])
        data.append((img, annotation))
    print("Processed: ", folder)
print(" ")        
        
df = pd.DataFrame(data=data, columns=['img_path', 'annotation_path'], index=None)
print("Total number of samples in the dataset: ", len(df))
print(" ")
df.head(10)


# ## Visualization
# Let's take some random samples from the dataset and plot some cats along with the facial keypoints 

# In[ ]:


# Plot some cats and respective annotations
f, ax = plt.subplots(3,2, figsize=(20,15))

# Get six random samples
samples = df.sample(6).reset_index(drop=True)

for i, sample in enumerate(samples.values):
    # Get the image path
    sample_img = df['img_path'][i]
    # Get the annotation path
    sample_annot = df['annotation_path'][i]
    # Read the annotation file
    f = open(sample_annot)
    points = f.read().split(' ')
    points = [int(x) for x in points if x!='']
    # Get the list of x and y coordinates
    xpoints = points[1:19:2]
    ypoints = points[2:19:2]
    # close the file
    f.close()
    
    ax[i//2, i%2].imshow(imread(sample_img))
    ax[i//2, i%2].axis('off')
    ax[i//2, i%2].scatter(xpoints, ypoints, c='g')
    
plt.show()    


# Next, we will try to build a Deep CNN that can learn the facial embeddings for a cat

# In[ ]:




