#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
import random
import math
import time, os
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt
from glob import glob

get_ipython().run_line_magic('matplotlib', 'inline')


# ## What is the purpose of this notebook?
# There is difference between train images and test images, I will do the following analysis.
# - visualizing train images and test images, comparing their distributions, found a few main differences
# - adding image augmentation methods onto train images
# - comparing augmented train images with test images

# ## visualize train_images

# In[ ]:


nrows, ncols = 4, 8

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20))

for i,fname in enumerate(np.random.choice(glob('../input/pku-autonomous-driving/train_images/*'), nrows*ncols)):
    img = Image.open(fname)
    axes[i//ncols, i%ncols].imshow(img)
    axes[i//ncols, i%ncols].axis('off')

plt.subplots_adjust(left=0.1, bottom=0.7, right=0.9, top=0.9,  wspace=0.01, hspace=0.1)


# ## visualize test_images
# Tune the random seed to randomly browse the test images

# In[ ]:


nrows, ncols = 4, 8
seed = np.random.randint(1, 2019)
print('seed=', seed)
np.random.seed(seed)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20))

for i,fname in enumerate(np.random.choice(glob('../input/pku-autonomous-driving/test_images/*'), nrows*ncols)):
    img = Image.open(fname)
    axes[i//ncols, i%ncols].imshow(img)
    axes[i//ncols, i%ncols].axis('off')
    axes[i//ncols, i%ncols].set_title(fname.split('/')[-1][:-4])

plt.subplots_adjust(left=0.1, bottom=0.7, right=0.9, top=0.9, wspace=0.01, hspace=0.2)


# In[ ]:


# img = cv2.imread('../input/pku-autonomous-driving/test_images/ID_23495df5e.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# fig = plt.figure(figsize=(8,8))
# plt.imshow(img)


# ## ...found four train-test differences
# The following are found in the test images and **NOT found in most of the train images**, which cound be manully added.
# 1. contrast
# 2. brightness
# 3. noise
# 4. TODO: color jittering

# ### 1. contrast

# In[ ]:


fname_list = ['../input/pku-autonomous-driving/test_images/ID_73094e19c.jpg', 
              '../input/pku-autonomous-driving/test_images/ID_d4cd0361c.jpg',
              '../input/pku-autonomous-driving/test_images/ID_ff99f8c4b.jpg', 
              '../input/pku-autonomous-driving/test_images/ID_23495df5e.jpg',  
             ]

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 10))

for i,ax in enumerate(axes):
    fname = fname_list[i]
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.axis('off')

plt.subplots_adjust(left=0.1, bottom=0.7, right=0.7, top=0.9, wspace=0.05, hspace=0.2)


# ### 2. brightness

# In[ ]:


fname_list = ['../input/pku-autonomous-driving/test_images/ID_349fe3d94.jpg', 
              '../input/pku-autonomous-driving/test_images/ID_ea29783ac.jpg',
              '../input/pku-autonomous-driving/test_images/ID_fec6066ea.jpg', 
              '../input/pku-autonomous-driving/test_images/ID_38aaf4faa.jpg',  
             ]

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 10))

for i,ax in enumerate(axes):
    fname = fname_list[i]
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.axis('off')

plt.subplots_adjust(left=0.1, bottom=0.7, right=0.7, top=0.9, wspace=0.05, hspace=0.2)


# ### 3. noise

# In[ ]:


fname_list = ['../input/pku-autonomous-driving/test_images/ID_b2d1b80e3.jpg', 
              '../input/pku-autonomous-driving/test_images/ID_3970b0062.jpg',
              '../input/pku-autonomous-driving/test_images/ID_b185c63d9.jpg', 
              '../input/pku-autonomous-driving/test_images/ID_08b7d6f4a.jpg',  
             ]

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 10))

for i,ax in enumerate(axes):
    fname = fname_list[i]
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.axis('off')

plt.subplots_adjust(left=0.1, bottom=0.7, right=0.7, top=0.9, wspace=0.05, hspace=0.2)


# ### 4. TODO: color jittering

# In[ ]:





# ## Add augmentations on train images

# ### 1. contrast

# In[ ]:


seed = np.random.randint(1, 2019)
print('seed=', seed)
np.random.seed(seed)

fname_list = np.random.choice(glob('../input/pku-autonomous-driving/train_images/*'), 4)

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 10))

for i,ax in enumerate(axes):
    fname = fname_list[i]
    img = Image.open(fname)
    ##add contrast here
    enh = ImageEnhance.Contrast(img)
    img_enh = enh.enhance(np.random.uniform(1.5, 2))#PIL.Image
    img_enh = np.asarray(img_enh)
    img_enh = (img_enh/255).astype('float32')
    ##plot
    ax.imshow(img_enh)
    ax.axis('off')

plt.subplots_adjust(left=0.1, bottom=0.7, right=0.7, top=0.9, wspace=0.05, hspace=0.2)


# ### 2. brightness

# In[ ]:


seed = np.random.randint(1, 2019)
print('seed=', seed)
np.random.seed(seed)

fname_list = np.random.choice(glob('../input/pku-autonomous-driving/train_images/*'), 4)

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 10))

for i,ax in enumerate(axes):
    fname = fname_list[i]
    img = Image.open(fname)
    ##add Brightness here
    enh = ImageEnhance.Brightness(img)
    img_enh = enh.enhance(np.random.uniform(0.5, 1.0))#PIL.Image
    img_enh = np.asarray(img_enh)
    img_enh = (img_enh/255).astype('float32')
    ##plot
    ax.imshow(img_enh)
    ax.axis('off')

plt.subplots_adjust(left=0.1, bottom=0.7, right=0.7, top=0.9, wspace=0.05, hspace=0.2)


# ### 3. add noise

# In[ ]:


def add_noise(image):
    """gauss noise"""
    row,col,ch= image.shape
    mean = 0
    var = np.random.random()*0.01 #0.001~0.01
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 1)
    return noisy


# In[ ]:


seed = np.random.randint(1, 2019)
print('seed=', seed)
np.random.seed(seed)

fname_list = np.random.choice(glob('../input/pku-autonomous-driving/train_images/*'), 4)

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 10))

for i,ax in enumerate(axes):
    fname = fname_list[i]
    img = plt.imread(fname)
    img = (img/255).astype('float32')
    ##add noise here
    img_enh = add_noise(img)#gauss
    ##plot
    ax.imshow(img_enh)
    ax.axis('off')

plt.subplots_adjust(left=0.1, bottom=0.7, right=0.7, top=0.9, wspace=0.05, hspace=0.2)


# ### What next?
# It is always helpful to have training data distribution similar to test data, hence we could try training with image augmentations from this notebook and see if it helps model generalization on the testset.
