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


plt.axis("off")
plt.imshow(bw_img,cmap="gray")
plt.show()


# In[ ]:


h, w = bw_img.shape


# In[ ]:


# define filters
GX = np.array([[1, 0, -1], 
                [1, 0, -1], 
                [1, 0, -1]])  
GY = np.array([[1, 1, 1],
                [0, 0, 0],
                [-1, -1, -1]])  


# In[ ]:


# define images with 0s
newGX = np.zeros((h, w))
newGY = np.zeros((h, w))
newG = np.zeros((h, w))


# In[ ]:


# offset by 1
for i in range(1, h - 1):
    for j in range(1, w - 1):
        GXGrad = (GX[0, 0] * bw_img[i - 1, j - 1]) +                          (GX[0, 1] * bw_img[i - 1, j]) +                          (GX[0, 2] * bw_img[i - 1, j + 1]) +                          (GX[1, 0] * bw_img[i, j - 1]) +                          (GX[1, 1] * bw_img[i, j]) +                          (GX[1, 2] * bw_img[i, j + 1]) +                          (GX[2, 0] * bw_img[i + 1, j - 1]) +                          (GX[2, 1] * bw_img[i + 1, j]) +                          (GX[2, 2] * bw_img[i + 1, j + 1])

        newGX[i - 1, j - 1] = abs(GXGrad)

        GYGrad = (GY[0, 0] * bw_img[i - 1, j - 1]) +                  (GY[0, 1] * bw_img[i - 1, j]) +                  (GY[0, 2] * bw_img[i - 1, j + 1]) +                  (GY[1, 0] * bw_img[i, j - 1]) +                  (GY[1, 1] * bw_img[i, j]) +                  (GY[1, 2] * bw_img[i, j + 1]) +                  (GY[2, 0] * bw_img[i + 1, j - 1]) +                  (GY[2, 1] * bw_img[i + 1, j]) +                  (GY[2, 2] * bw_img[i + 1, j + 1])

        newGY[i - 1, j - 1] = abs(GYGrad)

        # Gradient Edge
        grad=((GXGrad**2) + (GYGrad**2))**0.5  
        newG[i-1,j-1] = grad



# In[ ]:


plt.axis("off") 
plt.imshow(newG, cmap='gray')
plt.show()


# In[ ]:




