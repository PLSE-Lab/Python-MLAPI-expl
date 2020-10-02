#!/usr/bin/env python
# coding: utf-8

# # Stain deconvolution
# 
# There are a few modalities of images in the dataset. 
# The majority of images is actually greyscale broadcasted to 3 channels. 
# The rest of the images the *actual* rgb images are stained with hematoxylin and eosin (atleast to my knowledge).
# 
# One can use that information to do stain deconvolution on rgb images and transform the dataset so that all images have just 1 intensity channel. We will approach this problem in this notebook.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import numpy as np
import math
import matplotlib.pyplot as plt
from skimage.color import rgb2grey, rgb2hed
from skimage.exposure import rescale_intensity
from sklearn.externals import joblib


def plot_list(images=[], labels=[], n_rows=1):
    n_img = len(images)
    n_lab = len(labels)
    n_cols = math.ceil((n_lab+n_img)/n_rows)
    plt.figure(figsize=(12,10))
    for i, image in enumerate(images):
        plt.subplot(n_rows,n_cols,i+1)
        plt.imshow(image)
    for j, label in enumerate(labels):
        plt.subplot(n_rows,n_cols,n_img+j+1)
        plt.imshow(label, cmap='nipy_spectral')
    plt.show()


# In[2]:


sample_images = joblib.load('../input/sample_stained_not_stained_images.pkl')


# Let's take a look at some example images from the dataset

# In[3]:


plot_list(sample_images,n_rows=4)


# # Stained Image filter
# 
# Before we extract hematoxylin and eosin staining we need to first filter the rgb images. 
# Very simple approach is to do the following:

# In[4]:


def is_stained(img):
    red_mean, green_mean, blue_mean = img.mean(axis=(0, 1))
    if red_mean == green_mean == blue_mean:
        return False
    else:
        return True


# # Deconvolution
# Now that we have we will extract those h and e channels from images with the use of `rgb2hed` function. 
# 
# Let's see what it does:

# In[5]:


for img in sample_images:
    if is_stained(img):
        img_hed = rgb2hed(img)
        img_hematoxilin = img_hed[:,:,0]
        img_eosin = img_hed[:,:,1]
        img_dab = img_hed[:,:,2]

        plot_list([img, img_hematoxilin, img_eosin, img_dab])


# Now we can write a function that takes hematoxyli and eosin channels and combines them together.
# We will parametrize which channels user wants to use as well.

# In[6]:


def stain_deconvolve(img, mode='hematoxylin_eosin_sum'):
    img_hed = rgb2hed(img)
    if mode == 'hematoxylin_eosin_sum':
        h, w = img.shape[:2]
        img_hed = rgb2hed(img)
        img_he_sum = np.zeros((h, w, 2))
        img_he_sum[:, :, 0] = rescale_intensity(img_hed[:, :, 0], out_range=(0, 1))
        img_he_sum[:, :, 1] = rescale_intensity(img_hed[:, :, 1], out_range=(0, 1))
        img_deconv = rescale_intensity(img_he_sum.sum(axis=2), out_range=(0, 1))
    elif mode == 'hematoxylin':
        img_deconv = img_hed[:, :, 0]
    elif mode == 'eosin':
        img_deconv = img_hed[:, :, 1]
    else:
        raise NotImplementedError('only hematoxylin_eosin_sum, hematoxylin, eosin modes are supported')
    return img_deconv


# Let's see the results and compare how does this intensity differs from taking a simple greyscale.

# In[7]:


for img in sample_images:
    if is_stained(img):
        deconv = stain_deconvolve(img)
        grey = 1-rgb2grey(img)
        plot_list([img, grey, deconv])


# The difference is not huge but for some images, for instance the 3rd image we were able to extract a cleaner image with `stain_deconvolve` than with `greyscale`. 

# # Full pipeline
# If you would like to see how we plugged stain deconvolution into our pipeline go to [open solution](https://github.com/neptune-ml/open-solution-data-science-bowl-2018)
# 
# ![full open solution pipeline](https://gist.githubusercontent.com/jakubczakon/10e5eb3d5024cc30cdb056d5acd3d92f/raw/e85c1da3acfe96123d0ff16f8145913ee65e938c/full_pipeline.png)
# 
# The stain deconvolution step is defined in the `preprocessing.py` file:
# 
# ```python 
# 
# class StainDeconvolution(BaseTransformer):
#     def __init__(self, mode):
#         self.mode = mode
# 
#     def transform(self, X):
#         X_deconvoled = []
#         for x in X[0]:
#             x = from_pil(x)
#             if is_stained(x):
#                 x_deconv = (stain_deconvolve(x, mode=self.mode) * 255).astype(np.uint8)
#             else:
#                 x_deconv = (rgb2grey(x) * 255).astype(np.uint8)
#             x_deconv = to_pil(x_deconv)
#             X_deconvoled.append(x_deconv)
#         return {'X': [X_deconvoled]}
# ```
# 
# If you want to use our implementation just go for it!

# In[ ]:




