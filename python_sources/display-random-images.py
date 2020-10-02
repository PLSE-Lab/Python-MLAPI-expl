#!/usr/bin/env python
# coding: utf-8

# **Display Random Images**

# The NIH recently released over 100,000 anonymized chest x-ray images and their corresponding labels to the scientific community. 
# 
# https://lukeoakdenrayner.wordpress.com/2017/12/18/the-chestxray14-dataset-problems/
# 
# https://stanfordmlgroup.github.io/projects/chexnet/
# 

# *Step 1: Import Python Packages*

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from glob import glob
import random
import cv2
#print(os.listdir('../input/sample/images'))


# *Step 2: Define Helper Functions*

# In[2]:


multipleImages = glob('../input/sample/images/**')
def plotImages2():
    r = random.sample(multipleImages, 9)
    plt.figure(figsize=(20,20))
    plt.subplot(331)
    plt.imshow(cv2.imread(r[0])); plt.axis('off')
    plt.subplot(332)
    plt.imshow(cv2.imread(r[1])); plt.axis('off')
    plt.subplot(333)
    plt.imshow(cv2.imread(r[2])); plt.axis('off')
    plt.subplot(334)
    plt.imshow(cv2.imread(r[3])); plt.axis('off')
    plt.subplot(335)
    plt.imshow(cv2.imread(r[4])); plt.axis('off')
    plt.subplot(336)
    plt.imshow(cv2.imread(r[5])); plt.axis('off')
    plt.subplot(337)
    plt.imshow(cv2.imread(r[6])); plt.axis('off')
    plt.subplot(338)
    plt.imshow(cv2.imread(r[7])); plt.axis('off')
    plt.subplot(339)
    plt.imshow(cv2.imread(r[8])); plt.axis('off')


# *Step 3: Display Random Images*

# In[3]:


plotImages2()


# In[4]:


plotImages2()


# In[5]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()

