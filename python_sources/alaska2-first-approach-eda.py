#!/usr/bin/env python
# coding: utf-8

# # Introduction

# First of all I wanted to say that, as a French student, I'm very happy to find a competition hosted by a French University, it's not so frequent. Thanks to the organizers!

# ### Definition of steganography

# **Definition** : steganography is usually referred to as the techniques and methods that allow to hide information within an innocuous-like cover
# object. The resulting stego-object resembles, as much as possible,
# the original cover objet. Therefore it can be sent over an unsecured
# communication channel that may be subject to wiretapping by an
# eavesdropper. Nowadays, steganography has been mostly developed for digital images because of their massive presence over the
# Internet, the universally adopted JPEG compression scheme and
# its relative simplicity to be modified [[1]](https://hal-utt.archives-ouvertes.fr/hal-02542075/file/J_MiPOD_vPub.pdf)

# > The main difference with cryptography is that in cryptography you don't want the user to understand the message. In steganography, you don't want the user to even know that there is a message.

# Here's a short video from Computerphile about Steganography in general and how you can hide information in the pixels of an image. Very instructive if you don't know much about the field!

# <iframe width="891" height="501" src="https://www.youtube.com/embed/TWEXCYQKyDc" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# This image explains the basic principle of steganography : you change the one or two least significant bytes (LSB). This way, the global modification of the picture is undetectable, but you can carry much information!
# <img align="right" width="300" height="300" src="https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSWW1PzPeVFMp9F3IhRq4zCN5FP3DvZpI5DEmOsOgH0ud2OhLew&usqp=CAU">

# ## Load Packages

# In[ ]:


import pandas as pd 
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import random

from tqdm.notebook import tqdm
from collections import defaultdict

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


PATH = "/kaggle/input/alaska2-image-steganalysis/"
IMAGE_IDS = os.listdir(os.path.join(PATH, 'Cover'))
N_IMAGES = len(IMAGE_IDS)
ALGORITHMS = ['JMiPOD', 'JUNIWARD', 'UERD']


# In[ ]:


sample_submission = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))

print(f"Number of training samples: {len(os.listdir(os.path.join(PATH, 'Cover')))} images")
print(f"Number of samples to predict: {sample_submission.shape[0]} images")


# * Lets take a look at what the sample submission look like
# * Note here that the term "Label" can be misleading because for each image, we want to predict a score and not a Label!

# In[ ]:


sample_submission.head()


# ## First Simple EDA

# The "format" of the competition is strange because we are asked to predict the probability that an image was modified using a steganography algorithm, but there is no clear label. We are provided 75k images and their respective transformation with 4 different algorithms, but we don't know which algorithm may have been applied to each image in the test set. 

# In[ ]:


# Data Samples
n_rows = 3
samples = np.random.randint(0, N_IMAGES, size=n_rows ** 2)
f, ax = plt.subplots(n_rows, n_rows, figsize=(14, 10))

for j, sample_id in enumerate(samples):
    img = mpimg.imread(os.path.join(PATH, 'Cover', IMAGE_IDS[sample_id]))
    ax[j // n_rows, j % n_rows].imshow(img)

plt.tight_layout()


# Quite random pictures, isn't it?
# Seems like all photos are 512 x 512 x 3 : let's check that

# It would be a bit long to check that all 75k images are the same size, so we randomly sample 1000 of them. (If you know a faster way to load .jpg images, please ping me!)

# In[ ]:


for img_id in tqdm(random.choices(IMAGE_IDS, k=1000)):
    img = mpimg.imread(os.path.join(PATH, 'Cover', img_id))
    assert img.shape == (512, 512, 3)


# So we're now know that all images are **512 x 512 x 3**.

# ## Inspect images from crypted folders

# Someting natural now is to look at a picture in its different versions, e.g. with and without the different algorithms applied.

# In[ ]:


# Data Samples
n_rows = 3
samples = np.random.randint(0, N_IMAGES, size=n_rows)
samples_r = np.repeat(samples, repeats=n_rows)
f, ax = plt.subplots(n_rows, n_rows, figsize=(14, 10))

for j, sample_id in enumerate(samples_r):
    for algo in ['Cover'] + ALGORITHMS:
        img = mpimg.imread(os.path.join(PATH, algo, IMAGE_IDS[sample_id]))
        ax[j // n_rows, j % n_rows].imshow(img)

plt.tight_layout()


# Seems like we don't see much difference. But that makes sense, since encrypted information should not be distinguishable just by seing the picture. We will know take a look at the difference in pixels. 

# In[ ]:


# Data Samples
n_rows = 3
samples = np.random.randint(0, N_IMAGES, size=n_rows)

img = defaultdict(dict)
for sample_id in samples:
    for algo in ['Cover'] + ALGORITHMS:
        img[algo][sample_id] = mpimg.imread(os.path.join(PATH, algo, IMAGE_IDS[sample_id]))

f, ax = plt.subplots(n_rows, n_rows, figsize=(14, 10))
        
img_diff = defaultdict(dict)
for j, sample_id in enumerate(samples):
    for k, algo in enumerate(ALGORITHMS):
        diff_im = (img['Cover'][sample_id] - img[algo][sample_id]) % 255
        
        ax[(n_rows * j + k) // n_rows, (n_rows * j + k) % n_rows].imshow(diff_im)
        ax[(n_rows * j + k) // n_rows, (n_rows * j + k) % n_rows].set_title(f'Mean Pixel difference = {diff_im.mean() :.2f}')
        
        img_diff[algo][sample_id] = diff_im
        


# Again, we don't see many any message... That's because the hidden message is stored in the least significant bytes!

# ## Analysing Least-Significant Bytes (LSBs)

# > [Credits](https://github.com/ragibson/Steganography)

# In[ ]:


import os
from time import time
from PIL import Image


def lsb_filter(path, n=2):
    image = Image.open(path)

    mask = (1 << n) - 1  # sets first bytes to 0

    color_data = [(255 * ((channel[0] & mask) + (channel[1] & mask) + (channel[2] & mask)) // (3 * mask),) * 3 for channel in image.getdata()]

    image.putdata(color_data)
    
    return image


# In[ ]:


# Data Samples
n_rows = 4
samples = np.random.randint(0, N_IMAGES, size=n_rows)
f, ax = plt.subplots(n_rows, n_rows, figsize=(14, 10))

lsb_images = defaultdict(dict)

for j, sample_id in enumerate(samples):
    for k, algo in enumerate(['Cover'] + ALGORITHMS):
        img = lsb_filter(os.path.join(PATH, algo, IMAGE_IDS[sample_id]))
        
        lsb_images[algo][sample_id] = img
        ax[(n_rows * j + k) // n_rows, (n_rows * j + k) % n_rows].imshow(img)
        ax[(n_rows * j + k) // n_rows, (n_rows * j + k) % n_rows].set_title(algo)


# In[ ]:


# Data Samples
n_rows = 4
#samples = np.random.randint(0, N_IMAGES, size=n_rows)
f, ax = plt.subplots(n_rows, n_rows - 1, figsize=(14, 10))
        
#img_diff = defaultdict(dict)
for j, sample_id in enumerate(samples):
    for k, algo in enumerate(ALGORITHMS):
        diff_im = (np.asarray(lsb_images['Cover'][sample_id]) - np.asarray(lsb_images[algo][sample_id])) % 255
        
        ax[(n_rows * j + k) // n_rows, (n_rows * j + k) % n_rows].imshow(diff_im)
        ax[(n_rows * j + k) // n_rows, (n_rows * j + k) % n_rows].set_title(f'Mean Pixel difference = {diff_im.mean() :.2f}')
        
        img_diff[algo][sample_id] = diff_im
        


# Even on the LSBs we don't see significant differences. Let's continue digging!

# ### I will update this notebook continuously so stay tuned!

# In[ ]:




