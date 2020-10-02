#!/usr/bin/env python
# coding: utf-8

# References: https://www.kaggle.com/xhlulu/panda-resize-and-save-train-data

# Pipeline Step 1 -  Pre-Processing:
# 
# Step 1 Part A - Downsizing:
# Each of our training examples are huge, around 25,000 x 15,000 px. There is a lot of empty space in these training examples, and GPU usage is limited here on Kaggle. It will be useful to downsize the training images. We can then obtain several labeled tiles per image and train our model to output a Gleason grade per tile.

# In[ ]:


# let us now import some useful libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os # file directories
import openslide # accessing large images
import matplotlib.pyplot as plt # plotting figures
from PIL import Image # open and display images
import cv2 #computer vision library
from tqdm.notebook import tqdm # progress bar, tqdm shorthand for progress in Arabic 
import skimage.io #image processing
from skimage.transform import resize, rescale


# In[ ]:


# setting the main directory and loading the train CSV file
MAIN_DIR = '../input/prostate-cancer-grade-assessment'
train = pd.read_csv(os.path.join(MAIN_DIR, 'train.csv')).set_index('image_id')


# In[ ]:


# setting the directory where the images and masks are located
data_dir = os.path.join(MAIN_DIR, 'train_images/')
mask_dir = os.path.join(MAIN_DIR, 'train_label_masks/')
mask_files = os.listdir(mask_dir)


# In[ ]:


# set the path for the first image
img_id = train.index[0]
path = data_dir + img_id + '.tiff'


# In[ ]:


# Check the time it takes to open the image with two methods
get_ipython().run_line_magic('time', 'biopsy = openslide.OpenSlide(path)')
get_ipython().run_line_magic('time', 'biopsy_a = skimage.io.MultiImage(path)')


# In[ ]:


# Check the time it takes to resize an image and compare quality
# note theses tiff files are multi-level images, there are three levels of differing quality and hence size. We select the lowest quality level for resizing.
get_ipython().run_line_magic('timeit', 'img_a = biopsy.get_thumbnail(size=(512, 512))')
get_ipython().run_line_magic('timeit', 'img_b = resize(biopsy_a[-1], (512, 512))')
get_ipython().run_line_magic('timeit', 'img_c = cv2.resize(biopsy_a[-1], (512, 512))')
get_ipython().run_line_magic('timeit', 'img_d = Image.fromarray(biopsy_a[-1]).resize((512, 512))')



# We see that skimage is quickest to load an image and cv2 is fastest for resizing. We now check to ensure there is no difference in quality by resizing method.

# In[ ]:


biopsy = openslide.OpenSlide(path)
biopsy_a = skimage.io.MultiImage(path)
img_0 = biopsy.get_thumbnail(size=(512, 512))
img_1 = resize(biopsy_a[-1], (512, 512))
img_2 = cv2.resize(biopsy_a[-1], (512, 512))
img_3 = Image.fromarray(biopsy_a[-1]).resize((512, 512))

fig, axs = plt.subplots(2, 2, figsize=(15, 15))
fig.suptitle('Ensuring no variance in image quality by resize method')
axs[0,0].imshow(img_0)
axs[0,1].imshow(img_1)
axs[1,0].imshow(img_2)
axs[1,1].imshow(img_3)
plt.show()


# Since there appears to be no major difference across resizing methods we will go with the fastest cv2.resize. There are several interpolation methods for how cv2 resizes images let us quickly check these methods.

# In[ ]:


interpolations = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
fig, axs = plt.subplots(3, 2, figsize=(15,15))
fig.suptitle('Ensuring no variance in image quality by interpolation method')
for i in range(0,5):
    axs[0,0].imshow(cv2.resize(biopsy_a[-1], (512, 512), interpolation = cv2.INTER_NEAREST))
    axs[0,1].imshow(cv2.resize(biopsy_a[-1], (512, 512), interpolation = cv2.INTER_LINEAR))
    axs[1,0].imshow(cv2.resize(biopsy_a[-1], (512, 512), interpolation = cv2.INTER_AREA))
    axs[1,1].imshow(cv2.resize(biopsy_a[-1], (512, 512), interpolation = cv2.INTER_CUBIC))
    axs[2,0].imshow(cv2.resize(biopsy_a[-1], (512, 512), interpolation = cv2.INTER_LANCZOS4))
plt.show()


# There is no glaring deviation in quality so we can continue with default interpolation method. Let us now check times for saving our resized image.

# In[ ]:


get_ipython().run_line_magic('timeit', "Image.fromarray(img_2).save(img_id + '.png')")
get_ipython().run_line_magic('timeit', "cv2.imwrite(img_id+'.png', img_2)")


# cv2 is also the fastest method for saving. Let us now load our masks.

# In[ ]:


mask = skimage.io.MultiImage(mask_dir + mask_files[1])
img = skimage.io.MultiImage(data_dir + mask_files[1].replace("_mask", ""))
# check the shapes of lowest resolution layer
mask[-1].shape, img[-1].shape


# In[ ]:


# we set our save directory
save_dir = "/kaggle/train_images/"
os.makedirs(save_dir, exist_ok=True)


# In[ ]:


# we resize and save all our images, and use tqdm to give our progress
for img_id in tqdm(train.index):
    load_path = data_dir + img_id + '.tiff'
    save_path = save_dir + img_id + '.png'
    
    biopsy = skimage.io.MultiImage(load_path)
    img = cv2.resize(biopsy[-1], (512, 512))
    cv2.imwrite(save_path, img)


# In[ ]:


# same for masks
save_mask_dir = '/kaggle/train_label_masks/'
os.makedirs(save_mask_dir, exist_ok=True)

for mask_file in tqdm(mask_files):
    load_path = mask_dir + mask_file
    save_path = save_mask_dir + mask_file.replace('.tiff', '.png')
    
    mask = skimage.io.MultiImage(load_path)
    img = cv2.resize(mask[-1], (512, 512))
    cv2.imwrite(save_path, img)

