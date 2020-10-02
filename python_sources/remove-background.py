#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import matplotlib.pyplot as plt
import numpy as np


# Some images have a background that represents shades of the same color.

# In[95]:


# For Example

img_lst = ['10050ed12fbad46d.png', '100bbf5e832083d3.png']

i = 1
plt.figure(figsize=[5,5])
for img_name in img_lst:
    img = cv2.imread("../input/train/%s" % img_name)[...,[2, 1, 0]]
    plt.subplot(2, 1, i)
    plt.imshow(img)
    i += 1
plt.show()


# We can change it to white color by replacing colors of pixel of image boundaries to 255.

# In[99]:


def rem_bkg(img):
    y_size,x_size,col = img.shape
    
    for y in range(y_size):
        for r in range(1,6):
            col = img[y, x_size-r] 
            img[np.where((img == col).all(axis = 2))] = [255,255,255]
        for l in range(5):
            col = img[y, l] 
            img[np.where((img == col).all(axis = 2))] = [255,255,255]

    for x in range(x_size):
        for d in range(1,6):
            col = img[y_size-d, x] 
            img[np.where((img == col).all(axis = 2))] = [255,255,255]
        for u in range(5):
            col = img[u, x] 
            img[np.where((img == col).all(axis = 2))] = [255,255,255]
    
    return img


# In[100]:


# Images after removing backgound color

i = 1
plt.figure(figsize=[5,5])
for img_name in img_lst:
    img = cv2.imread("../input/train/%s" % img_name)[...,[2, 1, 0]]
    
    rem_bkg(img)    
    
    plt.subplot(2, 1, i)
    plt.imshow(img)
    i += 1
plt.show()  

