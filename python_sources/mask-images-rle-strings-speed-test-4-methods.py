#!/usr/bin/env python
# coding: utf-8

# # Acknowledgments
# 
# I've tried some of the kernels and my implementation for self-learning.  
# Reference of the kernels are bellow. Thanks everyone for great posts!
# 
# * [Fast, tested RLE](https://www.kaggle.com/stainsby/fast-tested-rle) by [Sam Stains](https://www.kaggle.com/stainsby)
# * [Fast Run Length Encode](https://www.kaggle.com/paulorzp/fast-run-length-encode) by [Paulo Pinto](https://www.kaggle.com/paulorzp)
# * [Even Faster Run Length Encoder](https://www.kaggle.com/hackerpoet/even-faster-run-length-encoder) by [Kevin H](https://www.kaggle.com/hackerpoet) and [jeffeverett](https://www.kaggle.com/jeffeverett)

# ## Summary
# * about 6-7 ms / image: [Fast, tested RLE](https://www.kaggle.com/stainsby/fast-tested-rle) by [Sam Stains](https://www.kaggle.com/stainsby)
# * about 180 ms / image: [Fast Run Length Encode](https://www.kaggle.com/paulorzp/fast-run-length-encode) by [Paulo Pinto](https://www.kaggle.com/paulorzp)
# * about 18- ms / image: [Even Faster Run Length Encoder](https://www.kaggle.com/hackerpoet/even-faster-run-length-encoder) by [Kevin H](https://www.kaggle.com/hackerpoet) and [jeffeverett](https://www.kaggle.com/jeffeverett)
# * about 7-8 ms / image: my code
# 
# I tried 1 - 100 image.  
# In this kernel's speed-test method, time-increases with the number of images.  
# 
# **I want to share if there is a faster way:)**

# ## Initalization

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import os
import re
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# In[2]:


# Get filenames list of image files
re_masks = re.compile('(^(.+?)_[0-9]+?)_mask\.gif$')
list_masks = [name for name in os.listdir('../input/train_masks/') if re_masks.search(name)]
list_masks.sort()
list_masks[0:5]


# In[3]:


# Check image loading
np_img_tmp = np.uint8(Image.open('../input/train_masks/' + list_masks[0]))
plt.imshow(np_img_tmp)
print('min:%s, max: %s' % (np_img_tmp.min(), np_img_tmp.max()))
np_img_tmp


# In[4]:


# Make speed-test dataset, shape = (N, 1280, 1918), N=100
list_np_dataset_nx1280x1918 = []
for img_mask in list_masks[0:100]:
    list_np_dataset_nx1280x1918.append(np.uint8(Image.open('../input/train_masks/' + img_mask)))
np_dataset_nx1280x1918 = np.array(list_np_dataset_nx1280x1918)
np_dataset_nx1280x1918.shape


# ## [Fast, tested RLE](https://www.kaggle.com/stainsby/fast-tested-rle) by [Sam Stains](https://www.kaggle.com/stainsby)

# In[ ]:


# REF: https://www.kaggle.com/stainsby/fast-tested-rle

def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of 
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask, 
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)


# In[ ]:


# For run like map-function
def mapfunc_img_to_rle_stainsby(np_img_mask_nx1280x1918):
    # Get RLE string from 1280 * 1918 image by N * 1280 * 1918 dataset
    return [rle_to_string(rle_encode(np_img_mask_1280x1918)) for np_img_mask_1280x1918 in np_img_mask_nx1280x1918]


# In[ ]:


# convert 1 images
mapfunc_img_to_rle_stainsby(np_dataset_nx1280x1918[0:1])


# In[ ]:


get_ipython().run_cell_magic('timeit', '-n20 -r5 -p5', '# convert 1 images\nmapfunc_img_to_rle_stainsby(np_dataset_nx1280x1918[0:1])')


# In[ ]:


get_ipython().run_cell_magic('timeit', '-n20 -r5 -p5', '# convert 10 images\nmapfunc_img_to_rle_stainsby(np_dataset_nx1280x1918[0:10])')


# ## [Fast Run Length Encode](https://www.kaggle.com/paulorzp/fast-run-length-encode) by [Paulo Pinto](https://www.kaggle.com/paulorzp)

# In[ ]:


# REF: https://www.kaggle.com/paulorzp/fast-run-length-encode

def rle (img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    bytes = np.where(img.flatten()==1)[0]
    runs = []
    prev = -2
    for b in bytes:
        if (b>prev+1): runs.extend((b+1, 0))
        runs[-1] += 1
        prev = b
    
    return ' '.join([str(i) for i in runs])


# In[ ]:


# For run like map-function
def mapfunc_img_to_rle_paulorzp(np_img_mask_nx1280x1918):
    # Get RLE string from 1280 * 1918 image by N * 1280 * 1918 dataset
    return [rle(np_img_mask_1280x1918) for np_img_mask_1280x1918 in np_img_mask_nx1280x1918]


# In[ ]:


# convert 1 images
mapfunc_img_to_rle_paulorzp(np_dataset_nx1280x1918[0:1])


# In[ ]:


get_ipython().run_cell_magic('timeit', '-n20 -r5 -p5', '# convert 1 images\nmapfunc_img_to_rle_paulorzp(np_dataset_nx1280x1918[0:1])')


# In[ ]:


get_ipython().run_cell_magic('timeit', '-n20 -r5 -p5', '# convert 10 images\nmapfunc_img_to_rle_paulorzp(np_dataset_nx1280x1918[0:10])')


# ## [Even Faster Run Length Encoder](https://www.kaggle.com/hackerpoet/even-faster-run-length-encoder) by [Kevin H](https://www.kaggle.com/hackerpoet) and [jeffeverett](https://www.kaggle.com/jeffeverett)
# 
# 

# In[ ]:


# REF: https://www.kaggle.com/hackerpoet/even-faster-run-length-encoder

def rle_kevin_and_jeffeverett(img):
    flat_img = img.flatten()
    flat_img = np.where(flat_img > 0.5, 1, 0).astype(np.uint8)
    flat_img = np.insert(flat_img, [0, len(flat_img)], [0, 0])

    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ix = np.where(starts)[0] + 1
    ends_ix = np.where(ends)[0] + 1
    lengths = ends_ix - starts_ix

    encoding = ''
    for idx in range(len(starts_ix)):
        encoding += '%d %d ' % (starts_ix[idx], lengths[idx])
    return encoding


# In[ ]:


# For run like map-function
def mapfunc_img_to_rle_kevin_and_jeffeverett(np_img_mask_nx1280x1918):
    # Get RLE string from 1280 * 1918 image by N * 1280 * 1918 dataset
    return [rle_kevin_and_jeffeverett(np_img_mask_1280x1918) for np_img_mask_1280x1918 in np_img_mask_nx1280x1918]


# In[ ]:


# convert 1 images
mapfunc_img_to_rle_kevin_and_jeffeverett(np_dataset_nx1280x1918[0:1])


# In[ ]:


get_ipython().run_cell_magic('timeit', '-n20 -r5 -p5', '# convert 1 images\nmapfunc_img_to_rle_kevin_and_jeffeverett(np_dataset_nx1280x1918[0:1])')


# In[ ]:


get_ipython().run_cell_magic('timeit', '-n20 -r5 -p5', '# convert 10 images\nmapfunc_img_to_rle_kevin_and_jeffeverett(np_dataset_nx1280x1918[0:10])')


# ## My code
# **numpy.diff() and numpy.where() is very powerful.**  
# TODO: buffer-overflow?

# In[7]:


def np_2d_img_to_str_rle(np_mask_img):
    np_mask_img_vec = np_mask_img.reshape(np_mask_img.shape[0] * np_mask_img.shape[1] )
    np_diff = np.diff(np_mask_img_vec)
    np_where_start = np.where(np_diff == 1)[0]
    np_where_end = np.where(np_diff == 255)[0] # -1 -> 255 in np.uint8, buffer-overflow
    start = np_where_start + 2
    end = np_where_end - start + 2
    list_output = ['%s %s' % (start[i], end[i]) for i in range(len(np_where_start))]
    str_output = ' '.join(list_output)
    return str_output

# For run like map-function
def mapfunc_np_2d_img_to_str_rle(np_img_mask_nx1280x1918):
    # Get RLE string from 1280 * 1918 image by N * 1280 * 1918 dataset
    return [np_2d_img_to_str_rle(np_img_mask_1280x1918) for np_img_mask_1280x1918 in np_img_mask_nx1280x1918]


# In[8]:


# convert 1 images
mapfunc_np_2d_img_to_str_rle(np_dataset_nx1280x1918[0:1])


# In[9]:


get_ipython().run_cell_magic('timeit', '-n20 -r5 -p5', '# convert 1 images\nmapfunc_np_2d_img_to_str_rle(np_dataset_nx1280x1918[0:1])')


# In[10]:


get_ipython().run_cell_magic('timeit', '-n20 -r5 -p5', '# convert 10 images\nmapfunc_np_2d_img_to_str_rle(np_dataset_nx1280x1918[0:10])')


# In[ ]:


get_ipython().run_cell_magic('timeit', '-n20 -r5 -p5', '# convert 10 images\nmapfunc_np_2d_img_to_str_rle(np_dataset_nx1280x1918)')
