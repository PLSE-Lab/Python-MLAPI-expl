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


# In[ ]:


#print(os.listdir("../input/flowers/flowers/sunflower"))


# AN RGB FILE IS A FILE WITH 3 CHANNELS: RED, BLUE AND GREEN EACH CHANNEL WITH A RANGE FROM 0 TO 255 (256 UNIQUE VALUES). THIS COMBINATION GIVE US A PRECISION OF 16,777,216 OF COLOR BY PIXEL (256X256X256)
# 
# I ALWAYS THINK THAT THIS IS OBSOLETE.  
# 
# THIS SCRIPT CONVERT A RGB (3 CHANNELS) TO RGB (1 CHANNEL) WITH SAME PRECISION.

# In[ ]:


from keras.preprocessing.image import load_img
a= np.array(load_img("../input//flowers/flowers/sunflower/8014734302_65c6e83bb4_m.jpg"))


# In[ ]:


print(a.shape)


# In[ ]:


import matplotlib.pyplot as plt
plt.imshow(a)


# In[ ]:


b=a[:,:,0]
c=a[:,:,1]*256
d=a[:,:,2]*(256*256)


# In[ ]:


e=b+c+d
print(e)


# In[ ]:


print(e.shape)
print(e.min(),e.max())


# THIS FILE (e) ONLY HAS ONE CHANNEL WITH PRECISION OF 16,777,216 
# 
# IF YOU USE THIS KIND OF FILE IN A NEURAL NETWORK, THE TRAINING IS FASTER BECAUSE THE PICTURES OF 3 CHANNELS WERE CONVERT TO 1 CHANNEL FILE.
