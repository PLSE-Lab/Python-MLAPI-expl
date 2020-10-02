#!/usr/bin/env python
# coding: utf-8

# # Faster Approach - RGB to HSV & CMYK
# 
# Those who are working in image processing domain with limited processing power must have faced issue of color coversion time consuming. Believe me, I have been working on such issues for more than a year now. And after checking in-built functions of skimage, I came up with my own functions with help of numba. 
# 
# For demonstration purpose I have taken a large image of 4000 x 3000 pixel from siim-isic-melanoma-classification competition.
# 
# Do comment if this is helpful. 
# 
# I know these snippets couldn't be the most optimized one. So if you have better/improved function, do share with me. 
# 
# **Below are function I created for RGB to HSV/CMYK or both**
# 
# I hope those with limited resource will be benefited from these code snippets.
# 
# Same script is copied on Github https://gist.github.com/vrajs5/3f540ef3bc84bc31ca49c11ba6b0d6ae

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
from skimage import io
import os
from numba import njit
from skimage import color

RGB_SCALE = 255
CMYK_SCALE = 100

@njit
def rgb_to_hsv_cmyk(img):
    dx, dy, dz = img.shape
    convData = np.zeros((dx, dy, 7))
    
    for i in range(dx):
        for j in range(dy):
            r, g, b = img[i,j]/RGB_SCALE
            
            mx = max(r, g, b)
            mn = min(r, g, b)
            df = mx-mn
            if mx == mn:
                h = 0
            elif mx == r:
                h = (60 * ((g-b)/df) + 360) % 360
            elif mx == g:
                h = (60 * ((b-r)/df) + 120) % 360
            elif mx == b:
                h = (60 * ((r-g)/df) + 240) % 360
            if mx == 0:
                s = 0
            else:
                s = (df/mx)*100
            v = mx*100
            convData[i,j,:3] = h, s, v
            
            
            if (r == 0) and (g == 0) and (b == 0):
                convData[i,j,3:] = 0, 0, 0, CMYK_SCALE
            else:
                c, m, y = 1 - r, 1 - g, 1 - b
                
                # extract out k [0, 1]
                min_cmy = min(c, m, y)
                divisor = (1 - min_cmy)
                c = (c - min_cmy) / divisor
                m = (m - min_cmy) / divisor
                y = (y - min_cmy) / divisor
                k = min_cmy

                # rescale to the range [0,CMYK_SCALE]
                convData[i,j,3:] = c * CMYK_SCALE, m * CMYK_SCALE, y * CMYK_SCALE, k * CMYK_SCALE
                
    return convData

@njit
def rgb_to_cmyk(img):
    dx, dy, dz = img.shape
    cmykData = np.zeros((dx, dy, 4))
    
    for i in range(dx):
        for j in range(dy):
            if img[i,j].any() ==  False:
                cmykData[i,j] = 0, 0, 0, CMYK_SCALE
            else:
                c, m, y = 1 - img[i,j] / RGB_SCALE
                
                # extract out k [0, 1]
                min_cmy = min(c, m, y)
                divosor = (1 - min_cmy)
                c = (c - min_cmy) / divosor
                m = (m - min_cmy) / divosor
                y = (y - min_cmy) / divosor
                k = min_cmy

                # rescale to the range [0,CMYK_SCALE]
                cmykData[i,j] = c * CMYK_SCALE, m * CMYK_SCALE, y * CMYK_SCALE, k * CMYK_SCALE

    return cmykData


@njit
def rgb_to_hsv(img):
    dx, dy, dz = img.shape
    hsvData = np.zeros((dx, dy, dz))
    
    for i in range(dx):
        for j in range(dy):
            r, g, b = img[i,j]/RGB_SCALE
            mx = max(r, g, b)
            mn = min(r, g, b)
            df = mx-mn
            if mx == mn:
                h = 0
            elif mx == r:
                h = (60 * ((g-b)/df) + 360) % 360
            elif mx == g:
                h = (60 * ((b-r)/df) + 120) % 360
            elif mx == b:
                h = (60 * ((r-g)/df) + 240) % 360
            if mx == 0:
                s = 0
            else:
                s = (df/mx)*100
            v = mx*100
            hsvData[i,j] = h, s, v
    return hsvData


# Taking image of size 4000 x 6000 pixel size.

# In[ ]:


masterInpath = '/kaggle/input/siim-isic-melanoma-classification/'
imgName = 'ISIC_7685852.jpg'
img = io.imread(masterInpath + 'jpeg/train/' + imgName)
print(img.shape)


# **skimage color conversion function performance.**

# In[ ]:


get_ipython().run_line_magic('timeit', 'color.rgb2hsv(img)')


# **Performance of custom function**
# 
# Only difference between two function is custom one will convert hsv to its scale of 360, 100, 100

# In[ ]:


get_ipython().run_line_magic('timeit', 'rgb_to_hsv(img)')


# Performance of custom cmyk function

# In[ ]:


get_ipython().run_line_magic('timeit', 'rgb_to_cmyk(img)')


# Performance of combine hsv & cmyk function.

# In[ ]:


get_ipython().run_line_magic('timeit', 'rgb_to_hsv_cmyk(img)')

