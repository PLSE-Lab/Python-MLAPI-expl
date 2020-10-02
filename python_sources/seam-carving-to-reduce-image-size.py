#!/usr/bin/env python
# coding: utf-8

# # Seam Carving to reduce image size - an alternative to resampling?
# 
# 
# ## Introduction
# The most challenging part of this competition for a hardware shy Kaggler is the sheer size of the image (512x512) and the fact that the features  cover only a couple of pixels. On the other hand, there are lot of voids in these images.
# 
# So the question is, can we get rid of the voids and reduce the image size. Seam Carving might be the answer.
# 
# ## Seam Carving
# 
# Seam carving (or liquid rescaling) is an algorithm for content-aware image resizing ([Avidan & Shamir, 2007](https://perso.crans.org/frenoy/matlab2012/seamcarving.pdf)).
# 
# Here, I show it with a working example from the training set.

# In[ ]:


import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
from skimage.filters import sobel,gaussian
from skimage.transform import seam_carve 
from keras.preprocessing.image import load_img


# In[ ]:


trainset = pd.read_csv("../input/train.csv", index_col="Id")


# In[ ]:


ix = 200
filters = ['green','blue','red','yellow']


# In[ ]:


image = np.sum([np.array(load_img('../input/train/{}_{}.png'.format(trainset.index[ix],k)))[:,:,0]/255 for k in filters],axis=0)


# A fast `seam_carve` function available in `skimage.transforms` only works for seams along one direction. So, we need to call it twice.
# 
# I have used here a `sobel` of the gaussian smoothed image as the energy map.
# 

# In[ ]:


eimage = sobel(gaussian(image,4))
imageh = seam_carve(image, eimage,'horizontal',100)
eimageh = sobel(gaussian(imageh,4))
finimage = seam_carve(imageh, eimageh,'vertical',100)


# In[ ]:


plt.figure(figsize=(20,10))
plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(finimage)


# It seems to work on certain images in the training set like the one above. 
# Although, if you look carefully, the features seems to be slightly cropped in certain parts.  
# 
# But, with a better idea, I hope it can be used for all the images.  I think this would be far better way of reducing the image size than resampling due to the fine structure of the features.

# In[ ]:




