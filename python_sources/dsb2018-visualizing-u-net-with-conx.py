#!/usr/bin/env python
# coding: utf-8

# # Overview
# The notebook uses the [CONX package](https://github.com/Calysto/conx) from [Douglas Blank](https://cs.brynmawr.edu/~dblank/) and the Calysto team to show how the basic U-Net architecture works on segmentation problems. We take the Kaggle Data Science Bowl for 2018 as an exciting example of real-world segmentation and study how our MiniUNET architecture handles the problem. This kernel/notebook is best run interactively (fork it and run it a in an interactive Kaggle Session or download the IPython notebook and run it using mybinder on 
# [![imagetool](https://img.shields.io/badge/launch-UNET_Demo-yellow.svg)](http://mybinder.org/v2/gh/Quantitative-Big-Imaging/conx/master?urlpath=%2Fapps%2Fseg_notebooks%2FUNetDemo.ipynb)
# 
# ## Note
# This is not 'real' U-Net since it does not have the proper cropping layers nor the correct size and depth (Ronneberger trained the original model using 512x512 images and having many more layers of max-pooling and upsampling).  The cropping layers are important as well since edges can skew the weights in the convolutions and cause the algorithm to converge slowly or with small enough windows incorrectly.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread


# # Loading Training Data
# Here we load the images using the tools from Keegil in his [kernel](https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277?scriptVersionId=2164855). It is a very simplified way of loading the data in and downsampling everything, it is probably worthwhile to investigate better approaches for handling high-resolution data well. As other kernels show the image sizes vary wildly. We also apply the normalizing [code](https://www.kaggle.com/kmader/normalizing-brightfield-stained-and-fluorescence) to make the images more consistent 

# In[9]:


import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import chain
from skimage.io import imshow, imread_collection, concatenate_images
from skimage.io import imread as imread_raw
import cv2
from skimage.transform import resize
from skimage.morphology import label
def imread(in_path, grid_size = 8): 
    in_rgb_img = imread_raw(in_path)
    if len(in_rgb_img.shape)>2:
        bgr = in_rgb_img[:,:,[2,1,0]] # flip r and b
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:,:,0]
    else:
        lab = in_rgb_img
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(grid_size,grid_size))
    out_img = np.expand_dims(clahe.apply(lab), -1)
    return 255-out_img if out_img.mean()>127 else out_img

# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1
TRAIN_PATH = '../input/stage1_train/'
TEST_PATH = '../input/stage1_test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]
# Get and resize train images and masks
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread_raw(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask

# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

print('Done!')


# In[10]:


t_img = X_train[1]/255.0
m_img = Y_train[1]
fig, (ax_img, ax_mask) = plt.subplots(1,2, figsize = (12, 6))
ax_img.imshow(np.clip(255*t_img, 0, 255).astype(np.uint8) if t_img.shape[2]==3 else t_img[:,:,0],
              interpolation = 'none', cmap = 'bone')
ax_mask.imshow(m_img[:,:,0], cmap = 'bone')


# In[11]:


import conx as cx


# In[12]:


net = cx.Network(name = "MiniUNet")
base_depth = 8


# In[13]:


net.add(cx.ImageLayer("input", (IMG_WIDTH, IMG_HEIGHT) , IMG_CHANNELS)) 
net.add(cx.BatchNormalizationLayer("bnorm"))
c2 = lambda i, j, act = "relu": cx.Conv2DLayer("conv_{}".format(i, j), j, (3, 3), padding='same', activation=act)
net.add(c2(0, base_depth))
net.add(c2(1, base_depth))
net.add(cx.MaxPool2DLayer("pool1", pool_size=(2, 2), dropout=0.25))
net.add(c2(2, 2*base_depth))
net.add(c2(3, 2*base_depth))
net.add(cx.MaxPool2DLayer("pool2", pool_size=(2, 2), dropout=0.25))
net.add(c2(4, 4*base_depth))
net.add(c2(5, 4*base_depth))
net.add(cx.UpSampling2DLayer("up2", size = (2,2)))
net.add(cx.ConcatenateLayer("cat2"))
net.add(c2(6, 2*base_depth))
net.add(c2(7, 2*base_depth))
net.add(cx.UpSampling2DLayer("up1", size = (2,2)))
net.add(cx.ConcatenateLayer("cat1"))
net.add(c2(8, 2*base_depth))
net.add(cx.Conv2DLayer("output", 1, (1, 1), padding='same', activation='sigmoid'));


# # Connections
# We have to connect all of the layers together in a U-Net style. The tricky part is the skip connections that skip over the max pooling layers and go directly to the concatenate to combine the higher resolution information with the lower resolution feature space

# In[14]:


net.connect('input', 'bnorm')
net.connect('bnorm', 'conv_0')
net.connect('bnorm', 'cat1')
net.connect('conv_0', 'conv_1')


# In[15]:


net.connect('conv_1', 'pool1')
net.connect('pool1', 'conv_2')
net.connect('conv_2', 'conv_3')
net.connect('conv_3', 'pool2')
net.connect('pool2', 'conv_4')
net.connect('conv_4', 'conv_5')
net.connect('conv_5', 'up2')
net.connect('up2', 'cat2')
net.connect('conv_3', 'cat2')
net.connect('cat2', 'conv_6')
net.connect('conv_6', 'conv_7')
net.connect('conv_7', 'up1')
net.connect('up1', 'cat1')
net.connect('cat1', 'conv_8')
net.connect('conv_8', 'output')


# In[16]:


net.compile(error="binary_crossentropy", optimizer="adam")


# In[17]:


net.picture(t_img, dynamic = True, rotate = True, show_targets = True, show_errors=True, scale = 1.0)


# In[18]:


net.dataset.clear()
ip_pairs = [(x/255.0,y*1.0) for x,y in zip(X_train, Y_train)]
net.dataset.append(ip_pairs)
net.dataset.split(0.25)


# In[20]:


net.propagate_to_image("conv_5", t_img, scale = 4)


# In[21]:


net.train(epochs=25, record=True)


# In[22]:


net.propagate_to_image("conv_5", t_img, scale = 5)


# In[23]:


net.picture(t_img, dynamic = True, rotate = True, show_targets = True, show_errors=True, scale = 1.25)


# In[24]:


net.dashboard()


# In[25]:


net.movie(lambda net, epoch: net.propagate_to_image("conv_5", t_img, scale = 3), 
                'mid_conv.gif', mp4 = False)


# In[26]:


net.movie(lambda net, epoch: net.propagate_to_image("conv_8", t_img, scale = 3), 
                'hr_conv.gif', mp4 = False)


# In[27]:


net.movie(lambda net, epoch: net.propagate_to_image("output", t_img, scale = 3), 
                'output.gif', mp4 = False)


# In[ ]:




