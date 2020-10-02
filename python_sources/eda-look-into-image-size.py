#!/usr/bin/env python
# coding: utf-8

# This EDA is mainly to investigate the different sizes of images. [ChewZY has made a great Kernel about this](https://www.kaggle.com/chewzy/eda-weird-images-with-new-updates) yet I did want to try some additional things on my side, both for the excercise of actually implementing them myself, and to investigate some issues I am having on my model.
# 
# (more content will most likely be added over the next few days, until the end of the competition)

# In[ ]:


import numpy as np
import pandas as pd
import cv2

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image

from collections import Counter

get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))


# Looking at the size of the datafrmes we have:

# In[ ]:


train_labels=pd.read_csv('../input/train.csv', dtype=str)
#Changing the attribute ids into lists instead of str seperated by a ' ' to be able to count them
train_labels['attribute_ids']=train_labels['attribute_ids'].str.split(' ')
test_labels=pd.read_csv('../input/sample_submission.csv', dtype=str)

print('train : \n', train_labels.head())
print('\ntest : \n', test_labels.head())

print('\ntrain shape: ', len(train_labels))
print('\ntest shape: ', len(test_labels))


# A quick look at the firt 5 labels, and the total length of the dataframe

# In[ ]:


labels = pd.read_csv('../input/labels.csv', dtype=str)
print('labels : ', '\n', labels.head())

print('\nlabels len :', len(labels))


# First look:
# - 109,237 train images
# - 7,7443 test images (keep in mind that the final submission will predict on 5x more)
# - There are 1103 different labels, across 'culture' and 'tag'
# 

# Getting a visual look at the images is always a good idea (keep in mind these are all 3 of the same style, and in no way representative of the entire dataset, which should always be remembered when looking at the top rows of a dataframe, or first couple of images).

# In[ ]:


# Let's show a few images:
for i in range(3):
    name_image=train_labels['id'][i]
    image = plt.imread('../input/train/'+name_image+'.png')
    plt.imshow(image)
    plt.show()


# I wanted to have a look at the distribution of size of the images in the train dataset. The first thing is to look at the distribtion across width and height.

# In[ ]:


#Let's take a look at the sizes of the images:

width_list = []
height_list = []
for i in range(len(train_labels)):
    name_image=train_labels['id'][i]
    with Image.open('../input/train/'+name_image+'.png') as img:
        width, height = img.size
        #print('width: {} \nheight: {}'.format(width, height))
        width_list.append(width)
        height_list.append(height)
        
average_width = sum(width_list)/len(width_list)
average_height = sum(height_list)/len(height_list)

print('average width: {} and height: {}'.format(average_width, average_height))

fig, ax =plt.subplots(1,2, figsize=(15, 8))

sns.distplot(width_list, ax=ax[0])
ax[0].set_title('Image width')
sns.distplot(height_list, ax=ax[1])
ax[1].set_title('Image height')
fig.show()


# The biggest images seem to go all the way to over 5500 pixels wide, and 7500 pixel tall! It might be worth taking a look at those images in more detail to see what they are of.
# 
# We can also note that the highest value for width is superior than for height, the y scale is different, so the visual comparaison of the two graphs should be done carefully.

# What I was most interested in was the general shape of the images rather than width or height individually.

# In[ ]:


image_ratio_list = [int(x)/int(y) for x,y in zip(height_list, width_list)]
mean_ratio = sum(image_ratio_list)/len(image_ratio_list)
print('mean ratio (height/width) of images is: ', mean_ratio)

plt.subplots(figsize=(20, 8))
sns.distplot(image_ratio_list, bins=100)
#plt.axvline(mean_ratio,color='orange', label='mean')
plt.axvline(x=1, color='red', label='x=1')
plt.title('image ratio (height/width) distribution')


# This graph shows the distribution of images ratios. We can clearly see that a bit above and below 1 (the red line) there are pics of values. This clearly means most of the images are of slight rectangular shape, meaning resizing into a square might not be the ideal option for the data generators when feeding the neural network. We can also see that there are images that are up to 25 times higher than wide. (in order to see the wider than high images, we would need to get the inverse of this ratio, as all these images get squeezed near zero).
