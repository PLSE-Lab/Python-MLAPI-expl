#!/usr/bin/env python
# coding: utf-8

# # APTOS 2019 - [Image Super Resolution](https://github.com/idealo/image-super-resolution)
# 
# ---
# 
# One of my first thoughts in this competition was to undertand the quality of the given images.
# 
# My first aim with this kernel was to improve images quality to make better prediction, but after a few days, I realized that I couldn't take much advantage as predictive power from this process. That's because the images resolution is already pretty high and when we process them we usually reduce it around 256x256.
# 
# However, I guess many laboratories can't have the same tools and, most probably, there are many hospitals around the world that can't afford taking those high quality images.
# 
# So, that's the goal of this kernel: 
# > _Improving Diabetic Retinopathy images using free and open source tools!_
# 
# The process will be:
# 
# **1. Start from a low resolution images**
# 
# **2. Increase their resolution with "Image Super Resolution" - [ISR](https://github.com/idealo/image-super-resolution) project**
# 
# **3. Compare the results with the original ones**
# 
# ---

# ## What is exactly Image Super Resolution project?
# 
# ![ISR Image](https://idealo.github.io/image-super-resolution/figures/butterfly.png)
# 
# **From their website:**
# 
# The goal of this project is to upscale and improve the quality of low resolution images.
# 
# This project contains Keras implementations of different Residual Dense Networks for Single Image Super-Resolution (ISR) as well as scripts to train these networks using content and adversarial loss components.
# 
# The implemented networks include:
# 
# - The super-scaling Residual Dense Network described in [Residual Dense Network for Image Super-Resolution](https://arxiv.org/abs/1802.08797) (Zhang et al. 2018)
# - The super-scaling Residual in Residual Dense Network described in [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219) (Wang et al. 2018)
# - A multi-output version of the Keras VGG19 network for deep features extraction used in the perceptual loss
# - A custom discriminator network based on the one described in [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) (SRGANS, Ledig et al. 2017)
# 
# Read the full documentation at: https://idealo.github.io/image-super-resolution/.
# 
# Docker scripts and Google Colab notebooks are available to carry training and prediction. Also, we provide scripts to facilitate training on the cloud with AWS and nvidia-docker with only a few commands.
# 
# ISR is compatible with Python 3.6 and is distributed under the Apache 2.0 license. We welcome any kind of contribution. If you wish to contribute, please see the Contribute section.
# 
# ![Arch](https://idealo.github.io/image-super-resolution/figures/RRDN.jpg)
# 
# ---
# 
# _NOTE:_
# 
# I Discovered this project following [Dat Tran](https://www.linkedin.com/in/dat-tran-a1602320/) on LinkedIn, you can always learn from his posts.
# 
# He and his team realized this amazing open source project that can be applied to many domains!
# 
# Here is the full team:
# 
# **Francesco Cardinale**, github: [cfrancesco](https://github.com/cfrancesco)
# 
# **Zubin John**, github: [valiantone](https://github.com/valiantone)
# 
# **Dat Tran**, github: [datitran](https://github.com/datitran)

# In[ ]:


# libraries
import numpy as np
import pandas as pd

import os
print(os.listdir("../input"))

import cv2
from PIL import Image
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Firstly, We are going so resize the images to Low Resolution.. Let's say 200x200
im_size = 200

# Then, we apply ISR and see the results


# # 2019 Competition Image

# In[ ]:


# I'm going to take just 2 images (new and old competition) and work on them

new_path = f"../input/aptos2019-blindness-detection/train_images/d7bc00091cfc.png"
new_image = cv2.imread(new_path)
new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
h, w, c = new_image.shape
print('Original Size: {}, {}'.format(w, h))
new_image = cv2.resize(new_image, (im_size,im_size))

fig = plt.figure(figsize=(10,10))
plt.imshow(new_image)


# # 2015 Competition Image

# In[ ]:


old_path = f"../input/diabetic-retinopathy-resized/resized_train/resized_train/22_left.jpeg"
old_image = cv2.imread(old_path)
old_image = cv2.cvtColor(old_image, cv2.COLOR_BGR2RGB)
h, w, c = old_image.shape
print('Original Size: {}, {}'.format(w, h))
old_image = cv2.resize(old_image, (im_size,im_size))

fig = plt.figure(figsize=(10,10))
plt.imshow(old_image)


# # Install ISR

# In[ ]:


get_ipython().system('pip install ISR')


# In[ ]:


get_ipython().system('wget https://github.com/idealo/image-super-resolution/raw/master/weights/sample_weights/rdn-C6-D20-G64-G064-x2/ArtefactCancelling/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5')
get_ipython().system('wget https://github.com/idealo/image-super-resolution/raw/master/weights/sample_weights/rdn-C6-D20-G64-G064-x2/PSNR-driven/rdn-C6-D20-G64-G064-x2_PSNR_epoch086.hdf5')
get_ipython().system('wget https://github.com/idealo/image-super-resolution/raw/master/weights/sample_weights/rdn-C3-D10-G64-G064-x2/PSNR-driven/rdn-C3-D10-G64-G064-x2_PSNR_epoch134.hdf5')
get_ipython().system('mkdir weights')
get_ipython().system('mv *.hdf5 weights')


# In[ ]:


# import model
from ISR.models import RDN


# In[ ]:


get_ipython().run_cell_magic('time', '', "rdn = RDN(arch_params={'C': 6, 'D':20, 'G':64, 'G0':64, 'x':2})\nrdn.model.load_weights('weights/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5')")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'new_image_isr = rdn.predict(np.asarray(new_image))\nold_image_isr = rdn.predict(np.asarray(old_image))')


# In[ ]:


# This is to compare to the images created with ISR so we can better see differences
old_image = cv2.resize(old_image,(im_size*2,im_size*2))
new_image = cv2.resize(new_image,(im_size*2,im_size*2))


# In[ ]:


f = plt.figure(figsize=(10,10))
f.suptitle("Old Competition Image: Resized", fontsize=16)
plt.imshow(old_image)
plt.show()


# In[ ]:


f = plt.figure(figsize=(10,10))
f.suptitle("Old Competition Image: ImageSuperResolution", fontsize=16)
plt.imshow(old_image_isr)
plt.show()


# In[ ]:


f = plt.figure(figsize=(10,10))
f.suptitle("New Competition Image: Resized", fontsize=16)
plt.imshow(new_image)
plt.show()


# In[ ]:


f = plt.figure(figsize=(10,10))
f.suptitle("New Competition Image: ImageSuperResolution", fontsize=16)
plt.imshow(new_image_isr)
plt.show()


# # Conclusions
# 
# We can easily notice the differences between the processed images and the resized ones.
# For instance, the outlines and veins details improved significantly!
# 
# What do you think? Could it be useful for hospitals and labs to save in super expensive machinery?
# 
# I would love to discuss further in comments
# 
# ---
# 
# **If you enjoyed the kernel, please consider upvoting it.  xD **
