#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import cv2
#plt.rcParams['figure.figsize'] = [25, 25]


# In[ ]:


img = cv2.imread('download.jpg',1)
img2 = cv2.imread('ronaldo.jpg',1)
#plt.imshow('ronaldo.jpg',0)
#cv2.imshow('download.jpg',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


# In[ ]:


#cv2.imshow('download.jpg',img)
#cv2.imshow('ronaldo.jpg',img2)


# In[ ]:


plt.imshow(img.reshape(img.shape[0], img.shape[1]), cmap=plt.cm.Greys)


# In[ ]:


edges = cv2.Canny(img,100,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])


# In[ ]:




