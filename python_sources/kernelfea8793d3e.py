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


# In[24]:


cd train_val2019


# In[29]:


get_ipython().system('ls Birds/202/0046f8c09d5d6acaa78baeffb2ba5c43.jpg')


# In[32]:


from matplotlib.image import imread
import numpy as np

data = imread('Birds/202/0046f8c09d5d6acaa78baeffb2ba5c43.jpg')



# In[33]:


data


# In[35]:


from PIL import Image

img = Image.fromarray(data, 'RGB')
img

