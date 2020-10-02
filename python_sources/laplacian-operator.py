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


import numpy as np
import cv2
import matplotlib.pyplot as plt


# In[ ]:


path="../input/edgedetection/"


# In[ ]:


name=os.listdir(path)


# In[ ]:


image_name=path+name[11]


# In[ ]:


image = cv2.cvtColor(cv2.resize(cv2.imread(image_name),(224,224)), cv2.COLOR_BGR2RGB)


# In[ ]:


plt.axis("off")
plt.imshow(image)
plt.show()


# In[ ]:


img = np.array(image)


# In[ ]:


img.shape


# In[ ]:


bw_img =  cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


# In[ ]:


blur_img = cv2.GaussianBlur(bw_img, (3, 3), 0)


# In[ ]:


# Positive Laplacian Operator
laplacian = cv2.Laplacian(blur_img, cv2.CV_64F)


# In[ ]:


plt.axis("off")
plt.imshow(laplacian,cmap="gray")
plt.show()


# In[ ]:




