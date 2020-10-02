#!/usr/bin/env python
# coding: utf-8

# In this kernel I will be covering the basics of Computer vision.Topics like converting an image to Numpy array and opening it using PIL package.Using Open CV to do image processing.This kernel is work in process.If you like my work please do vote.

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


# **Importing the modules**

# In[ ]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image 


# **Loading the image**

# In[ ]:


pic=Image.open('..//input/L1_out.png')


# In[ ]:


pic


# This is picture of famous Lourve Museum in Paris 

# **Converting the picture into array**

# In[ ]:


type(pic)


# Lets convert the image to numpy array

# In[ ]:


pic_arr=np.asarray(pic)


# In[ ]:


type(pic_arr)


# Now the picture is converted to numpy array

# In[ ]:


pic_arr.shape


# In[ ]:


plt.imshow(pic_arr)
plt.ioff()


# imshow is used to show the images which are transformed into arrays

# **Making a copy of the array image **

# In[ ]:


pic_red=pic_arr.copy()


# In[ ]:


plt.imshow(pic_red)
plt.ioff()


# **Lets Zero our Contribution from Green and Blue Channels**

# In[ ]:


pic_red.shape


# **Displaying each color channels R G B**

# In[ ]:


# R G B 
plt.imshow(pic_red[:,:,0])   # O Stands for Red
plt.ioff()


# In[ ]:


plt.imshow(pic_red[:,:,1])    # 1 stands for Green
plt.ioff()


# In[ ]:


plt.imshow(pic_red[:,:,2])   # 2 stands for blue
plt.ioff() 


# **Displaying the individual color channel in grey scale**

# In[ ]:


# RED Channel value 0 means no red,Pure black 
#value of 255 means full read and will appear white in image

plt.imshow(pic_red[:,:,0],cmap='gray')   # 2 stands for blue
plt.ioff()


# In[ ]:


plt.imshow(pic_red[:,:,1],cmap='gray')   # 2 stands for blue
plt.ioff()


# In[ ]:


plt.imshow(pic_red[:,:,2],cmap='gray')   # 2 stands for blue
plt.ioff()


# **Making the contribution of Green and Blue to Zero**

# In[ ]:


# Making the green channel zero 
pic_red[:,:,1]=0


# In[ ]:


# Making the blue channel zero 
pic_red[:,:,2]=0


# In[ ]:


plt.imshow(pic_red)
plt.ioff()


# This picture has contribution of only red channel

# **Now we will be Open CV package **

# In[ ]:


img=cv2.imread('..//input/L1_out.png')


# If you provide a wrong path no errot will be displayed

# In[ ]:


type(img)


# With Open CV there is no need to convert he image to numpy to process it

# In[ ]:


img.shape


# In[ ]:


plt.imshow(img)
plt.ioff()


# The image is appearing more blue as Matplotlib and Open CV have different order for processing color channel
# 
# Matplotlib --> RGB RED GREEN BLUE
# 
# Open CV    --> BGR BLUE GREEN RED

# **Converting the image from BGR To RGB**

# In[ ]:


fix_image=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


# In[ ]:


plt.imshow(fix_image)
plt.ioff()


# **Reading an image as Gray Scale**

# In[ ]:


img_gray=cv2.imread('..//input/L1_out.png',cv2.IMREAD_GRAYSCALE)


# In[ ]:


img_gray.max()


# In[ ]:


plt.imshow(img_gray)
plt.ioff()


# In[ ]:


plt.imshow(img_gray,cmap='gray')
plt.ioff()


# In[ ]:


plt.imshow(img_gray,cmap='magma')
plt.ioff()


# **Resizing Image**

# In[ ]:


fix_image.shape


# In[ ]:


new_image=cv2.resize(fix_image,(1000,500))


# In[ ]:


plt.imshow(new_image)
plt.ioff()


# **Resizing the image based on ratio**

# In[ ]:


w_ratio=0.5
h_ratio=0.5


# In[ ]:


new_image_1=cv2.resize(fix_image,(0,0),fix_image,w_ratio,h_ratio)


# In[ ]:


plt.imshow(new_image_1)
plt.ioff()


# In[ ]:


new_image_1.shape


# **Flipping the image of Horizontal or Vertical **

# In[ ]:


new_img_f=cv2.flip(fix_image,0)
plt.imshow(new_img_f)
plt.ioff()


# In[ ]:


new_img_f=cv2.flip(fix_image,1)
plt.imshow(new_img_f)
plt.ioff()


# In[ ]:


new_img_f=cv2.flip(fix_image,-1)
plt.imshow(new_img_f)
plt.ioff()


# **Saving a image file **

# In[ ]:


#cv2.imwrite('',fix_image)


# **Adjusting figure Size **

# In[ ]:


fig=plt.figure(figsize=(10,8))
ax=fig.add_subplot(111)
ax.imshow(fix_image)
plt.ioff()


# In[ ]:




