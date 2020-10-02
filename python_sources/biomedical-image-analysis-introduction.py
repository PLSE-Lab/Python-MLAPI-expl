#!/usr/bin/env python
# coding: utf-8

# This kernel is introduction of Biomedical Image Analysis. I am learning this topic from datacamp so this kernel is just for learning purpose.
# 
# **Introduction** 
# 
# Since first X-ray image in 1895, Medical Imaging technology advanced a lot. 
# 
# ![](https://upload.wikimedia.org/wikipedia/commons/e/e3/First_medical_X-ray_by_Wilhelm_R%C3%B6ntgen_of_his_wife_Anna_Bertha_Ludwig%27s_hand_-_18951222.gif)
# 
# 
# **Loading Image**
# 
# We will use imageio python package to read images. Imageio is a Python library that provides an easy interface to read and write a wide range of image data, including animated images, volumetric data, and scientific formats(https://pypi.org/project/imageio/).

# In[ ]:


import imageio
import os


# In[ ]:


os.listdir('../input')


# Let's load first image from train set only

# In[ ]:


first_image = os.listdir('../input/stage_1_train_images')[0]


# In[ ]:


img = imageio.imread('../input/stage_1_train_images/4ba3e640-eb0a-4f4f-900c-af7405bc1790.dcm')
im  = imageio.imread('body-001.dcm')


# In[ ]:




