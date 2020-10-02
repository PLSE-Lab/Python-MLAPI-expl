#!/usr/bin/env python
# coding: utf-8

# In[19]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
from PIL import Image
import matplotlib.pyplot as plt


# # Exploring the directory of Subject 01 and clothes 06
# For kaggle kernels the data is in **'../input/somaset/somaset/01/06/'**

# In[20]:


get_ipython().system("ls '../input/somaset/somaset/01/06/'")


# # Visualize the image #26.

# In[22]:


img_array = np.array(Image.open('../input/somaset/somaset/01/06/0026.jpg'))
plt.imshow(img_array)


# In[ ]:




