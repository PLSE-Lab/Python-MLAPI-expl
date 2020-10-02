#!/usr/bin/env python
# coding: utf-8

# In[34]:


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


# In[36]:


# Import the required libraries
import cv2                         # opencv version 3.4.2
import numpy as np                 # numpy version 1.16.3
import matplotlib.pyplot as plt    # matplotlib version 3.0.3
get_ipython().run_line_magic('matplotlib', 'inline')


# In[37]:


# Load the source image
src_img = cv2.imread('../input/harita 7/harita2.png')


# In[38]:


img_src = cv2.imread('../input/harita 2/harita2',0)
 print img_src


# In[40]:


# Function to display an image using matplotlib
def show_image(img, title, colorspace):
    dpi = 72
    figsize = (img.shape[1] / dpi, img.shape[0] / dpi)
    fig, ax = plt.subplots(figsize = figsize, dpi = dpi)
    if colorspace == 'RGB':
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), interpolation = 'spline16')
    if colorspace == 'gray':
        plt.imshow(img, cmap = 'gray')
    plt.title(title, fontsize = 12)
    ax.axis('off')
    plt.show()    


# In[41]:


# Display the source image
show_image(src_img, 'harita2', 'RGB')


# In[42]:


# Change colorspace from BGR to HSV
src_img_hsv = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV)

# Define limits of yellow HSV values
red_lower = np.array([240, 50, 50])
red_upper = np.array([255, 0, 0])

# Filter the image and get the mask
mask = cv2.inRange(src_img_hsv, red_lower, red_upper)

show_image(mask, 'black color filter mask', 'gray')


# In[45]:


# Importing Libraries
import numpy as np # For number crunching
import pandas as pd # For organizing data
import matplotlib.pyplot as plt # For visualization


# In[44]:


import pandas as pd


# In[46]:


df = pd.read_csv('../input/building-footprints/LI_BUILDING_FOOTPRINTS (1).csv', index_col=['OBJECTID'])


# In[47]:


df.head(3)


# In[48]:


df.iloc[3]


# In[49]:


df.Shape__Area.iloc[[0]]


# In[51]:


short=input("Short Edge : ")
long=input("Lond Edge : ")
s=int(short)
l=int(long)
area=s*l
circumference=2*(s*1)
print("Area:",area)
print("circumference",circumference)


# In[52]:


short=input("Short Edge : ")
long=input("Long Edge : ")
s=int(short)
l=int(long)
Yapilasma=(s-5)*(l-5)
print("Yapilasma Alani:",Yapilasma)


# In[53]:


Alan=input("Alan : ")
Yukseklik=input("Yukseklik : ")
s=int(Alan)
l=int(Yukseklik)
Oto=(s*l)/25
print("Gereken Otopark Adedi:",Oto)


# In[54]:


Alan=input("Alan : ")
s=int(Alan)
Circ=s*(15/100)
print("Ortalama Sirkulasyon Alani:",Circ)

