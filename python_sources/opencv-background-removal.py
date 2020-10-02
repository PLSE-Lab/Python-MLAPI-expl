#!/usr/bin/env python
# coding: utf-8

# ## Fruits 360 backgound removal
# 
# In this kernel you can find easy and ready-to-use way for removing backgound from images. I used OpenCV, PIL and numpy.
# 
# <b>Motivation</b>:<br>
# Actually, the main thing in Data Scientist's job, it is data processing. But what can you do, if there is no ready dataset? You can create it by yourself! Doing this, sometimes, you need to concatenate images. So, there it is ;)
# 
# ***
# Contents:
# * [Imports](#Imports)
# 
# * [First look on images](#First-look-on-images)
# 
# * [Background removal](#Background-removal)
# 
# * [Result](#Result)
# 
# ***
# 
# #### Acknowledgments
# 
# 1. [Mihai Oltean and his team](https://www.kaggle.com/moltean) - for damn great dataset.
# 2. [OpenCV](https://opencv.org) - for joy and pain. 

# ## Imports

# In[ ]:


import os
import random

import numpy as np
import cv2

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm
from PIL import Image 


# In[ ]:


DATA_DIR = '/kaggle/input/fruits/fruits-360_dataset/fruits-360/'
TEST_DATA_DIR = os.path.join(DATA_DIR, 'Test')

os.listdir(DATA_DIR)


# In[ ]:


fruits_list = os.listdir(TEST_DATA_DIR)
fruits_list[:5]


# ## First look on images

# In[ ]:


def get_random_fruit_image():
    fruit_class = np.random.choice(list(fruits_list), size=1)[0]
    fruit_class_path = os.path.join(TEST_DATA_DIR, fruit_class)

    image_names = os.listdir(fruit_class_path)
    image_name = np.random.choice(image_names, size=1)[0]

    image_path = os.path.join(fruit_class_path, image_name)
    image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image, fruit_class


# In[ ]:


image, fruit_class = get_random_fruit_image()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.title(f'Fruit class: {fruit_class}')
plt.imshow(image)


# Let's plot more!

# In[ ]:


fig, axs = plt.subplots(5, 6, figsize=(20, 15))
axs = axs.flatten()

for ax in axs:
    image, fruit_class = get_random_fruit_image()

    ax.set_title(f'Fruit class: {fruit_class}')
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ax.imshow(image)
    
plt.tight_layout()


# Background is the same everywhere, it is good.

# ## Background removal

# For this task we are using `cv2.threshold` function from OpenCV. You can read more about it [here](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html). From documentation:
# 
# > First argument is the source image, which should be a grayscale image. Second argument is the threshold value which is used to classify the pixel values. Third argument is the maxVal which represents the value to be given if pixel value is more than (sometimes less than) the threshold value.

# In[ ]:


def remove_background(img, threshold):
    """
    This method removes background from your image
    
    :param img: cv2 image
    :type img: np.array
    :param threshold: threshold value for cv2.threshold
    :type threshold: float
    :return: RGBA image
    :rtype: np.ndarray
    """
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshed = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

    cnts = cv2.findContours(morphed, 
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[0]

    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    mask = cv2.drawContours(threshed, cnt, 0, (0, 255, 0), 0)
    masked_data = cv2.bitwise_and(img, img, mask=mask)

    x, y, w, h = cv2.boundingRect(cnt)
    dst = masked_data[y: y + h, x: x + w]

    dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(dst_gray, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(dst)

    rgba = [r, g, b, alpha]
    dst = cv2.merge(rgba, 4)

    return dst


# In[ ]:


fig, axs = plt.subplots(3, 2, figsize=(12, 8))
fig.suptitle('Background removal result', fontsize=14)

for ax in axs:
    image, fruit_class = get_random_fruit_image()
    
    ax[0].set_title(f'Fruit class: {fruit_class}. Original image.')
    ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    image = remove_background(image, threshold=250.)
    
    ax[1].set_title(f'Fruit class: {fruit_class}. With no backgound.')
    ax[1].imshow(image)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# In[ ]:


def create_blank_image(height, width, rgb_color=(0, 0, 255)):
    """
    Creates np.array, each channel of which is filled with value from rgb_color
    
    Was stolen from:
    :source: https://stackoverflow.com/questions/4337902/how-to-fill-opencv-image-with-one-solid-color
    """
    
    image = np.zeros((height, width, 3), np.uint8)
    color = tuple(reversed(rgb_color))
    
    image[:] = color
    
    return image


# In[ ]:


fig, axs = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle('Blank images', fontsize=14)

axs = axs.flatten()

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

for idx, ax in enumerate(axs):
    image = create_blank_image(100, 100, rgb_color=COLORS[idx])
    
    ax.set_title(f'Blank image with color: {COLORS[idx]}')
    ax.imshow(image)


# In[ ]:


def add_backgound(image):
    """
    Adds background to given image using PIL
    """
    
    image_shape = image.shape
    image_height = image_shape[0]
    image_width = image_shape[1]
    
    backgound = create_blank_image(image_height, 
                                   image_width,
                                   rgb_color=(0, 0, 255))
    
    background = Image.fromarray(backgound)
    image = Image.fromarray(image)
    
    background.paste(image,
                     (0, 0),
                     image)
    
    return background


# ## Result

# In[ ]:


fig, axs = plt.subplots(6, 3, figsize=(16, 12))
fig.suptitle('Background adding result', fontsize=14)

for ax in axs:
    image, fruit_class = get_random_fruit_image()
    
    ax[0].set_title(f'Fruit class: {fruit_class}. Original image.')
    ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    image = remove_background(image, threshold=250.)
    
    ax[1].set_title(f'Fruit class: {fruit_class}. With no backgound.')
    ax[1].imshow(image)
    
    image = add_backgound(image)

    ax[2].set_title(f'Fruit class: {fruit_class}. With new backgound.')
    ax[2].imshow(image)
    
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# You can play with `threshold` value by yourself and find which suits more to your dataset. Good luck!
