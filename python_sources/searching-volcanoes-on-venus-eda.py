#!/usr/bin/env python
# coding: utf-8

# ## Exploratory Analysis

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd
import cv2
import seaborn as sns


# In[ ]:


#Paths for training and test
input_dir  = '../input'
train_dir = input_dir + '/volcanoes_train'
test_dir = input_dir + '/volcanoes_test'


# In[ ]:


# Removing index column and header = none
train_images = pd.read_csv(train_dir + '/train_images.csv',index_col=False,header=None)
train_labels = pd.read_csv(train_dir + '/train_labels.csv')


# In[ ]:


# Count plot on volcano presence in train image
sns.countplot(x = 'Volcano?',data=train_labels)


# In[ ]:


#count plot on no. of volcanos if there is a volcanoes in a image
sns.countplot(x = 'Number Volcanoes',data=train_labels)


# #### Note : As per above countplots, classes are imbalanced 

# ### Sample image and corresponding labels

# In[ ]:


image_sample = np.reshape(train_images.iloc[9].values,(110,110))
plt.imshow(image_sample,cmap='gray')
plt.title(train_labels.iloc[0])


# #### Taking 4 sample images with and without volcano

# In[ ]:


image_ids_with_volcanos = train_labels[train_labels['Volcano?']==1].iloc[0:5].index
image_ids_without_volcanos = train_labels[train_labels['Volcano?']==0].iloc[0:5].index
image_ids_with_volcanos


# In[ ]:


def display_images(ids,gaussin_blur=False,median_blur=False):
    columns = 5
    rows = 1
    fig=plt.figure(figsize=(12, 12))
    indx = 0
    for i in range(1, columns*rows +1):
        img = np.uint8(np.reshape(train_images.iloc[ids[indx]].values,(110,110)))
        if gaussin_blur:
            img = cv2.GaussianBlur(img,(5,5),0)
        if median_blur:
            img = cv2.medianBlur(img,9)
        fig.add_subplot(rows, columns, i)
        plt.title('type : ' + str(train_labels.iloc[ids[indx]].Type))
        plt.imshow(img,cmap='gray')        
        indx = indx + 1
    plt.show() 


# In[ ]:


display_images(image_ids_with_volcanos)


# In[ ]:


display_images(image_ids_with_volcanos, gaussin_blur=True)


# In[ ]:


display_images(image_ids_with_volcanos, median_blur=True)


# In[ ]:


display_images(image_ids_without_volcanos)


# In[ ]:


display_images(image_ids_without_volcanos,gaussin_blur=True)


# In[ ]:


display_images(image_ids_without_volcanos,median_blur=True)


# ## Conclusion
# 
# #### Classes are imbalance
# If you observe first count plot, it is clear that count of images without volcanoes is almost 7x of images with volcanoes.
# #### Low resolution single channel
# Images are having 110x110 resolution.
# #### Images looks better with gaussian blur/median blur
# Noise filtering gives better visibility of volcanoes in image

# In[ ]:




