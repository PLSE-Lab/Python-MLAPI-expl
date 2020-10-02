#!/usr/bin/env python
# coding: utf-8

# # Morphological Transformations & Custom albumentations transforms
# This kernel shows some image augmentation using morphological transformations.
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
# 
# ---
# 
# **This kernel fork from [yu4u's Bengali: morphological ops as image augmentation](https://www.kaggle.com/ren4yu/bengali-morphological-ops-as-image-augmentation).**  
# **if you like this kernel, please upvote original kernels.**

# In[ ]:


# from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from albumentations import Compose
from albumentations.core.transforms_interface import ImageOnlyTransform


# In[ ]:


perquet_path = "/kaggle/input/bengaliai-cv19/train_image_data_0.parquet"
df = pd.read_parquet(perquet_path)
h = 137
w = 236


# In[ ]:


def get_augmented_img(img, func):
    output_img = np.zeros((h * 2, w), dtype=np.uint8)
    output_img[:h] = img
    output_img[h:] = func(img)
    return output_img


# In[ ]:


def show_augmented_img(f):
    cols, rows = 5, 3
    img_num = cols * rows
    fig = plt.figure(figsize=(18,12))

    for i in range(img_num):
        img = get_augmented_img(data[i], f)
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(img)
        ax.set_axis_off()


# In[ ]:


sub_df = df.sample(n=15)
data = 255 - sub_df.iloc[:, 1:].values.reshape(-1, h, w).astype(np.uint8)


# # Erosion

# In[ ]:


def erosin(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(np.random.randint(1, 6, 2)))
    img = cv2.erode(img, kernel, iterations=1)
    return img

show_augmented_img(erosin)


# ### Custom albumentations transform

# In[ ]:


class Erosin(ImageOnlyTransform):
    def apply(self, img, **params):
        return erosin(img)

f = Erosin()
out = f(image=data[0])

plt.imshow(out["image"])


# # Dilation

# In[ ]:


def dilation(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(np.random.randint(1, 6, 2)))
    img = cv2.dilate(img, kernel, iterations=1)
    return img

show_augmented_img(dilation)


# ### Custom albumentations transform

# In[ ]:


class Dilation(ImageOnlyTransform):
    def apply(self, img, **params):
        return dilation(img)
    
    
f = Dilation()
out = f(image=data[0])

plt.imshow(out["image"])


# # Opening
# Opening is performed by successively applying erosion and dilation. It can be used to remove noise. Here erosion and dilation with random kernel is used as image augmentation.

# In[ ]:


def get_random_kernel():
    structure = np.random.choice([cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS])
    kernel = cv2.getStructuringElement(structure, tuple(np.random.randint(1, 6, 2)))
    return kernel

def opening(img):
    img = cv2.erode(img, get_random_kernel(), iterations=1)
    img = cv2.dilate(img, get_random_kernel(), iterations=1)
    return img

show_augmented_img(opening)


# ### Custom albumentations transform

# In[ ]:


class Opening(ImageOnlyTransform):
    def apply(self, img, **params):
        return opening(img)
    
f = Opening()
out = f(image=data[0])

plt.imshow(out["image"])


# # Closing
# Closing is performed by successively applying dilation and erosion. It can be used to 'close' small holes. Here dilation and erosion with random kernel is used as image augmentation.

# In[ ]:


def closing(img):
    img = cv2.dilate(img, get_random_kernel(), iterations=1)
    img = cv2.erode(img, get_random_kernel(), iterations=1)
    return img

show_augmented_img(closing)


# ### Custom albumentations transform

# In[ ]:


class Closing(ImageOnlyTransform):
    def apply(self, img, **params):
        return closing(img)
    
    
f = Closing() 
out = f(image=data[0])

plt.imshow(out["image"])


# ## albumentations.Compose using Custom albumentations transforms

# In[ ]:


transform = Compose([
    Erosin(),
    Dilation(),
    Opening(),
    Closing()
])

out = transform(image=data[0])
plt.imshow(out["image"])

