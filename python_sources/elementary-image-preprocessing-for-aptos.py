#!/usr/bin/env python
# coding: utf-8

# 
# ## Purpose of this kernel
# 
# Hi, I've been interested in image processing world for a long time, and finally joined APTOS competition.
# 
# Same as other novice competitors, I begun to read discussions and kernels witten by other competitors, and I especially enjoyed Michael Kazachok's nice kernel (https://www.kaggle.com/miklgr500/auto-encoder), which demonstrate auto encoder technique and apply PCA on image data, because his kernel gave me great insight about how dataset are distributed.
# 
# 
# In adittion, I learned elementary techniques to preprocess image data from Kazachok's kernel, so write this kernel which introduce how to read and preprocess images for newcomers like me.
# 
# Any comments for clarification and correction are welcome.

# ## Import packages

# In[ ]:


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Read image
# 
# Use cv2(OpenCV) package's imread() to read images.

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


image_id = (train_df['id_code'])[1]
path = f"../input/train_images/{image_id}.png"
img = cv2.imread(path)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) ## note that cv.imread() gives BGR array, so need convert it to RGB array


# ## Convert image into gray-scale
# 
# Convert a BGR image into a gray-scale image by cv2.cvtColor().
# Because better performances are achieved with more information in general, I have no confidence that I should always discard information of colors. However, at least there are several reasons which justify to discard them:
# 
# 1. As mentioned by the organizer, the images were taken with various kinds of cameras, so color information could work as noise. ("The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.")
# 
# 2. Images in this dataset are not so colorful, and it's likely that imformation of colors are not so extensively useful for this competitions. (Though I have no perfect confidence about it, because I'm not a doctor!)
# 
# 3. Discard colors can reduce data size and could save computing time.

# In[ ]:


img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.gray()
plt.imshow(img)


# ## Clip and resize
# 
# Cut off outside of retina image, and resize it into handy size.

# In[ ]:


tol = 5
mask = img > tol

nz_rows = mask.any(1)  ## filter rows where all values are less than tol out
nz_cols = mask.any(0)  ## filter cols where all values are less than tol out

img = img[np.ix_(nz_rows, nz_cols)]
plt.imshow(img)


# In[ ]:


img = cv2.resize(img, (224, 224))
plt.imshow(img)


# ### Sharpen image
# Sharpen the image by subtracting blured image, and add mean level.
# 
# In fact, I don't understand how magic numbers (used for cv2.GaussianBlur() and cv2.addWeighted()) are determined.
# Anyway, you can find the basic concept of this process (i.e., using Gaussian Blur as low-pass filter) in various articles (the URL below, for example).
# 
# https://stackoverflow.com/questions/4993082/how-to-sharpen-an-image-in-opencv

# In[ ]:


kernel_size = (0, 0)
sigma_XY = 224/10
img2 = cv2.GaussianBlur(img, kernel_size, sigma_XY)
plt.imshow(img2)


# In[ ]:


img = cv2.addWeighted(img, 4, img2, -4, 128)  ## img = 4 * img - 4 * img2 + 128
plt.imshow(img)  ## sharpened image

