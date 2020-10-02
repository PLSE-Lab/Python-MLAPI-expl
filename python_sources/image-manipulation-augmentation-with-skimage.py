#!/usr/bin/env python
# coding: utf-8

# # Image transformation / augmentation with skimage Python library
# 
# This notebook aims to gather various ways to manipulate images in order to proceed to dataset augmentation.
# 
# ### Table of content
# 
# * [Rescale image](#rescale)
# * [Add random noise](#random-noise)
# * [Color color inversion](#color-inversion)
# * [Rotate image](#rotate)
# * [Rescale color intensity (change contrast)](#rescale-intensity)
# * [Gamma correction](#change-gamma)
# * [Logarithmic correction](#log-correction)
# * [Sigmoid correction](#sigmoid-correction)
# 
# ## <a id='rescale'></a> Rescale image

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import rescale
import warnings
warnings.filterwarnings("ignore")

original_image = data.chelsea()
image_rescaled = rescale(original_image, 1.0 / 4.0)

def show_images(before, after, op):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    ax = axes.ravel()
    ax[0].imshow(before, cmap='gray')
    ax[0].set_title("Original image")

    ax[1].imshow(after, cmap='gray')
    ax[1].set_title(op + " image")
    if op == "Rescaled":
        ax[0].set_xlim(0, 400)
        ax[0].set_ylim(300, 0)
    plt.tight_layout()

show_images(original_image, image_rescaled, "Rescaled")


# ## <a id='random-noise'></a> Add random noise

# In[ ]:


from skimage.util import random_noise

image_with_random_noise = random_noise(original_image)

show_images(original_image, image_with_random_noise, "Random noise")


# ## <a id='gray-scale'></a> Color to gray scale

# In[ ]:


from skimage.color import rgb2gray

gray_scale_image = rgb2gray(original_image)

show_images(original_image, gray_scale_image, "Gray scale")


# ## <a id='color-inversion'></a> Image color inversion

# In[ ]:


from skimage import util

color_inversion_image = util.invert(original_image)

show_images(original_image, color_inversion_image, "Inversion image")


# ## <a id='zoom'></a> Rotate image

# In[ ]:


from skimage.transform import rotate

# perform a 45 degree rotation
image_with_rotation = rotate(original_image, 45)

show_images(original_image, image_with_rotation, "Rotated")


# ## <a id='rescale-intensity'></a> Rescale intensity (change contrast)

# In[ ]:


import numpy as np
from skimage import exposure

v_min, v_max = np.percentile(original_image, (0.2, 99.8))
better_contrast = exposure.rescale_intensity(original_image, in_range=(v_min, v_max))

show_images(original_image, better_contrast, 'Rescale intensity')


# ## <a id='change-gamma'></a> Gamma correction

# In[ ]:


# gamma and gain parameters are between 0 and 1
adjusted_gamma_image = exposure.adjust_gamma(original_image, gamma=0.4, gain=0.9)

show_images(original_image, adjusted_gamma_image, 'Adjusted gamma')


# ## <a id='log-correction'></a> Logarithmic correction

# In[ ]:


log_correction_image = exposure.adjust_log(original_image)

show_images(original_image, log_correction_image, 'Logarithmic corrected')


# ## <a id='sigmoid-correction'></a> Sigmoid correction

# In[ ]:


sigmoid_correction_image = exposure.adjust_sigmoid(original_image)

show_images(original_image, log_correction_image, 'Sigmoid corrected')

