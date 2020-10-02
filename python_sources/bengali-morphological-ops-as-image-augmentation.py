#!/usr/bin/env python
# coding: utf-8

# # Morphological Transformations
# This kernel shows some image augmentation using morphological transformations.
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html

# In[ ]:


from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


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


def f(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(np.random.randint(1, 6, 2)))
    img = cv2.erode(img, kernel, iterations=1)
    return img

show_augmented_img(f)


# # Dilation

# In[ ]:


def f(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(np.random.randint(1, 6, 2)))
    img = cv2.dilate(img, kernel, iterations=1)
    return img

show_augmented_img(f)


# # Opening
# Opening is performed by successively applying erosion and dilation. It can be used to remove noise. Here erosion and dilation with random kernel is used as image augmentation.

# In[ ]:


def get_random_kernel():
    structure = np.random.choice([cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS])
    kernel = cv2.getStructuringElement(structure, tuple(np.random.randint(1, 6, 2)))
    return kernel

def f(img):
    img = cv2.erode(img, get_random_kernel(), iterations=1)
    img = cv2.dilate(img, get_random_kernel(), iterations=1)
    return img

show_augmented_img(f)


# # Closing
# Closing is performed by successively applying dilation and erosion. It can be used to 'close' small holes. Here dilation and erosion with random kernel is used as image augmentation.

# In[ ]:


def get_random_kernel():
    structure = np.random.choice([cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS])
    kernel = cv2.getStructuringElement(structure, tuple(np.random.randint(1, 6, 2)))
    return kernel

def f(img):
    img = cv2.dilate(img, get_random_kernel(), iterations=1)
    img = cv2.erode(img, get_random_kernel(), iterations=1)
    return img

show_augmented_img(f)

