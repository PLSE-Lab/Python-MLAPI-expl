#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# *This is a part of practice on image processing to get familiar with different image processing techniques*. Python has a wonderful package of cv2 inspired from MATLAB to aid computer vision , Step by step we will progress towards more deep concepts once we are done with basics of image preprocessing .This is inspired by a vlog in Analytics Vidya. Here I have selected a random picture and have applied basic steps like pixel transformation , backgroung blur to develop basics of image processing 

# In[1]:


import cv2


# In[2]:


im = cv2.imread("../input/tanker.jpg")


# In[3]:


print(im)


# In[4]:


print(cv2.imshow)


# In[5]:


print(im.shape)


# In[ ]:


im[:, :, (0, 1)] = 0


# In[6]:


from matplotlib import pyplot as plt
from skimage.color import rgb2gray 


# In[ ]:


plt.imshow(im)


# In[ ]:


red , yellow = im.copy(),im.copy()


# In[ ]:


print(red)


# In[ ]:


print(yellow)


# In[ ]:


red[:,:,(1,2)]=0


# In[ ]:


plt.imshow(red)


# In[ ]:


yellow[:,:,2]=0


# In[ ]:


plt.imshow(yellow)


# In[ ]:


print(im.shape)


# In[9]:


gray = rgb2gray(im)


# In[10]:


gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


# In[11]:


print(gray.shape)


# *code to do the transformation*

# In[ ]:


from skimage.filters import threshold_otsu


# In[ ]:


thresh = threshold_otsu(gray)


# In[ ]:


binary = gray > thresh


# In[ ]:


plt.imshow(binary)


# In[ ]:


print(binary.shape)


# In[12]:


from skimage.filters import gaussian_filter


# In[13]:


blur_image = gaussian_filter(gray, sigma = 20)


# In[14]:


print(blur_image)


# In[15]:


plt.imshow(blur_image)


# In[ ]:




