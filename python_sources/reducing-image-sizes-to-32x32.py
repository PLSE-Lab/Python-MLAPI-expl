#!/usr/bin/env python
# coding: utf-8

# # Reducing Image Sizes to 32x32
# 
# I think that some of you will be interested in trying smaller models to get started (e.g. a CNN with only a few connected layers). However, those datasets seem to be really big (150k test images and 195k training images) as well as high resolution. Just trying to create a GPU kernel and preprocessing the images seem to take a while. 
# 
# Therefore, I created this kernel in order to reduce the image to the smallest usable size (i.e. 32x32, similar to CIFAR10/100). Please feel free to use this as an output to your exploration models, or to modify this for other image sizes.
# 
# Let me know your thoughts!
# 
# ### References
# * https://www.kaggle.com/xhlulu/exploration-and-preprocessing-for-keras-224x224

# In[ ]:


import os
import cv2
import math

import numpy as np # linear algebra
from PIL import Image
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


# ## Exploration

# In[ ]:


label_df = pd.read_csv('../input/train.csv')
submission_df = pd.read_csv('../input/sample_submission.csv')
label_df.head()


# In[ ]:


label_df['category_id'].value_counts()[1:16].plot(kind='bar')


# In[ ]:


def display_samples(df, columns=4, rows=3):
    fig=plt.figure(figsize=(5*columns, 3*rows))

    for i in range(columns*rows):
        image_path = df.loc[i,'file_name']
        image_id = df.loc[i,'category_id']
        img = cv2.imread(f'../input/train_images/{image_path}')
        fig.add_subplot(rows, columns, i+1)
        plt.title(image_id)
        plt.imshow(img)

display_samples(label_df)


# ## Preprocessing

# In[ ]:


def get_pad_width(im, new_shape, is_rgb=True):
    pad_diff = new_shape - im.shape[0], new_shape - im.shape[1]
    t, b = math.floor(pad_diff[0]/2), math.ceil(pad_diff[0]/2)
    l, r = math.floor(pad_diff[1]/2), math.ceil(pad_diff[1]/2)
    if is_rgb:
        pad_width = ((t,b), (l,r), (0, 0))
    else:
        pad_width = ((t,b), (l,r))
    return pad_width

def pad_and_resize(image_path, dataset, pad=False, desired_size=32):
    img = cv2.imread(f'../input/{dataset}_images/{image_path}.jpg')
    
    if pad:
        pad_width = get_pad_width(img, max(img.shape))
        padded = np.pad(img, pad_width=pad_width, mode='constant', constant_values=0)
    else:
        padded = img
    
    resized = cv2.resize(padded, (desired_size,)*2).astype('uint8')
    
    return resized


# ## Pad and resize all the images

# In[ ]:


get_ipython().run_cell_magic('time', '', "train_resized_imgs = []\ntest_resized_imgs = []\n\nfor image_id in label_df['id']:\n    train_resized_imgs.append(\n        pad_and_resize(image_id, 'train')\n    )\n\nfor image_id in submission_df['Id']:\n    test_resized_imgs.append(\n        pad_and_resize(image_id, 'test')\n    )")


# In[ ]:


X_train = np.stack(train_resized_imgs)
X_test = np.stack(test_resized_imgs)

target_dummies = pd.get_dummies(label_df['category_id'])
train_label = target_dummies.columns.values
y_train = target_dummies.values

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)


# ## Saving

# In[ ]:


# No need to save the IDs of X_test, since they are in the same order as the 
# ID column in sample_submission.csv
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)

