#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, 
import cv2
# Input data files are available in the "../input/" directory
import os
import matplotlib.pyplot as plt
import itertools
# import segmentation_models as sm
import keras
import random
# from iteration_utilities import unique_everseen
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


# In[ ]:


train_dir = '/kaggle/input/recursion-cellular-image-classification/'
os.listdir('/kaggle/input/recursion-cellular-image-classification/')


# In[ ]:


df1 = pd.read_csv(train_dir + 'pixel_stats.csv')
df1.head(10)


# In[ ]:


df2 = pd.read_csv(train_dir + 'test.csv')
df2.head(10)


# In[ ]:


df3 = pd.read_csv(train_dir + 'test_controls.csv')
df3.head(10)


# In[ ]:


df4 = pd.read_csv(train_dir + 'train_controls.csv')
df4.head(10)


# In[ ]:


df4.info()


# In[ ]:


df5 = pd.read_csv(train_dir + 'train.csv')
df5.tail(10)


# In[ ]:


df5.info()


# In[ ]:


os.listdir(train_dir + 'train/')


# In[ ]:


im = cv2.imread(train_dir + 'train/' + 'HUVEC-15/Plate3/C16_s2_w4.png')
plt.imshow(im)


# In[ ]:




