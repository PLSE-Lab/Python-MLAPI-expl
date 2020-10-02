#!/usr/bin/env python
# coding: utf-8

# # Pixel Art
# 
# ## Pixel art is a kind of visual we saw on old game console. I love them so much.
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import cv2
import matplotlib.pyplot as plt
import os


# In[ ]:


def pixelator(image):

    scale_percent = 40 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    return resized


# In[ ]:


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_UNCHANGED)
        if img is not None:
            images.append(img)
    return images


# In[ ]:


collection = load_images_from_folder('/kaggle/input/anime-faces/data/')


# # Take an example

# In[ ]:


for i in range(120,130):
    plt.figure()
    plt.imshow(cv2.cvtColor(collection[i],cv2.COLOR_BGR2RGB))


# In[ ]:


result = []

for i in range(len(collection)):
    #img = cv2.imread(collection[i], cv2.IMREAD_UNCHANGED)
    output = pixelator(collection[i])
    result.append(output)


# # The Result

# In[ ]:


for j in range(120,130):
    plt.figure()
    plt.imshow(cv2.cvtColor(result[j],cv2.COLOR_BGR2RGB))


# In[ ]:




