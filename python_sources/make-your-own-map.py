#!/usr/bin/env python
# coding: utf-8

# **Make Your Own Map**
# 
# I really like how the list of cities corresponds to a picture. I figured I would try making my own list of cities that forms a picture and share it with Kaggle.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2
import os
print(os.listdir("../input"))
from tqdm import tqdm

# Any results you write to the current directory are saved as output.


# First load the picture. I found it from this link.
# 
# https://i.ytimg.com/vi/6rdzpvBvCJE/hqdefault.jpg

# In[ ]:


img = cv2.imread("../input/cat-image/download.jpg", cv2.IMREAD_GRAYSCALE)


# In[ ]:


plt.imshow(img, cmap='Greys_r');


# The image looks really nice because it is in greyscale, but we will have to set everything to either black or white. For my code below, I am using a value of "80" as the cutoff. Might get better results by playing around with this value.

# In[ ]:


img.shape


# In[ ]:


cutoff = 80
cityid = 0
cities = np.array([-1, 0, 0]).reshape(1, 3)
for i, row in tqdm(enumerate(img)):
    for j, col in enumerate(row):
        if col > cutoff:
            cities = np.concatenate((cities, np.array([cityid, j, img.shape[1] - i]).reshape(1,3)))
            cityid +=1


# In[ ]:


cities = cities[1:]
cities = pd.DataFrame(cities)
cities.columns = ['cityId', 'X', 'Y']


# In[ ]:


fig = plt.figure(figsize=(4.8, 3.6) )
ax = fig.gca()
ax.set_facecolor('Black')
ax.set_xticks([])
ax.set_yticks([])
plt.scatter(cities.X, cities.Y, color='White', marker=".", alpha=.1);


# Don't try to submit this file XD

# In[ ]:


cities.to_csv('cat.csv')


# In[ ]:





# In[ ]:




