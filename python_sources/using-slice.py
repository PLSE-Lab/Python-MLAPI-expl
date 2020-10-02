#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

target_size = (400,400)

def show(img):
    plt.imshow(img)
    plt.show()

def get_img(path, color_space=cv2.COLOR_BGR2RGB):
    img = cv2.imread(path)
    if target_size:
        img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, color_space)
    return img

def select_channel(img, ch):
    return np.stack((img[:,:,ch],img[:,:,ch],img[:,:,ch])).transpose((1,2,0))


# In[ ]:


from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import glob

n_segments = 3
sigma = 5
for i, filepath in enumerate(glob.glob('../input/train/Type_1/*.jpg')):
    if i != 6:
        continue
    img = get_img(filepath, color_space=cv2.COLOR_BGR2RGB)

    R = select_channel(img, 0)
    G = select_channel(img, 1)
    B = select_channel(img, 2)
    segmentsR = slic(R, n_segments=n_segments, sigma=sigma)
    #segmentsG = slic(G, n_segments=n_segments, sigma=sigma)
    #segmentsB = slic(B, n_segments=n_segments, sigma=sigma)

    mark_segments = get_img(filepath, color_space=cv2.COLOR_BGR2RGB)
    mark_segments = mark_boundaries(mark_segments, segmentsR)
    #mark_segments = mark_boundaries(mark_segments, segmentsG)
    #mark_segments = mark_boundaries(mark_segments, segmentsB)

    show(mark_segments)


# In[ ]:


roi = img[100:300,:350,:]
show(roi)

scjB = slic(B[100:300,:350,:], n_segments=4, sigma=3)
scjG = slic(G[100:300,:350,:], n_segments=4, sigma=3)
mark_segments = roi.copy()
mark_segments = mark_boundaries(roi, scjB)
mark_segments = mark_boundaries(roi, scjG)
show(mark_segments)


# In[ ]:




