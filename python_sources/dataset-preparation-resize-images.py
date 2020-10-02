#!/usr/bin/env python
# coding: utf-8

# <a id="toc"></a>
# # Table of Contents
# 1. [Configure parameters](#configure_parameters)
# 1. [Import modules](#import_modules)
# 1. [Define helper-functions](#define_helper_functions)
# 1. [Update train.csv](#update_train_csv)
# 1. [Get a copy of sample_submission.csv](#get_a_copy_of_sample_submission_csv)
# 1. [Resize images](#resize_images)
#   1. [Resize train images](#resize_train_images)
#   1. [Resize test images](#resize_test_images)
# 1. [Zip new train_images and test_images directories](#zip_new_train_images_and_test_images_directories)

# <a id="configure_parameters"></a>
# # Configure parameters
# [Back to Table of Contents](#toc)

# In[ ]:


DATASET_DIR = '/kaggle/input/understanding_cloud_organization/'
ORI_SIZE = (1400, 2100) # (height, width)
NEW_SIZE = (384, 576) # (height, width)

import cv2
INTERPOLATION = cv2.INTER_CUBIC


# <a id="import_modules"></a>
# # Import modules
# [Back to Table of Contents](#toc)

# In[ ]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm_notebook


# <a id="define_helper_functions"></a>
# # Define helper-functions
# [Back to Table of Contents](#toc)

# In[ ]:


def mask2rle(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(mask_rle, shape):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in
                       (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


# <a id="update_train_csv"></a>
# # Update train.csv
# [Back to Table of Contents](#toc)

# In[ ]:


df = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'))


# In[ ]:


for idx, row in df.iterrows():
    encodedpixels = row[1]
    if encodedpixels is not np.nan:
        mask = rle2mask(encodedpixels, shape=ORI_SIZE[::-1])
        mask = cv2.resize(mask, NEW_SIZE[::-1], interpolation=INTERPOLATION)

        rle = mask2rle(mask)
        df.at[idx, 'EncodedPixels'] = rle


# In[ ]:


df.to_csv('./train.csv', index=False)


# <a id="get_a_copy_of_sample_submission_csv"></a>
# # Get a copy of sample_submission.csv
# [Back to Table of Contents](#toc)

# In[ ]:


cp $DATASET_DIR/sample_submission.csv ./


# <a id="resize_images"></a>
# # Resize images
# [Back to Table of Contents](#toc)

# <a id="resize_train_images"></a>
# ## Resize train images
# [Back to Table of Contents](#toc)

# In[ ]:


get_ipython().system('mkdir /kaggle/train_images')


# In[ ]:


train_images_dir = os.path.join(DATASET_DIR, 'train_images')
image_files = os.listdir(train_images_dir)

for image_file in tqdm_notebook(image_files):
    img = cv2.imread(os.path.join(train_images_dir, image_file))
    img = cv2.resize(img, NEW_SIZE[::-1], interpolation=INTERPOLATION)

    dst = os.path.join('/kaggle/train_images', image_file)
    cv2.imwrite(dst, img)


# <a id="resize_test_images"></a>
# ## Resize test images
# [Back to Table of Contents](#toc)

# In[ ]:


get_ipython().system('mkdir /kaggle/test_images')


# In[ ]:


test_images_dir = os.path.join(DATASET_DIR, 'test_images')
image_files = os.listdir(test_images_dir)

for image_file in tqdm_notebook(image_files):
    img = cv2.imread(os.path.join(test_images_dir, image_file))
    img = cv2.resize(img, NEW_SIZE[::-1], interpolation=INTERPOLATION)

    dst = os.path.join('/kaggle/test_images', image_file)
    cv2.imwrite(dst, img)


# <a id="zip_new_train_images_and_test_images_directories"></a>
# # Zip new train_images and test_images directories
# [Back to Table of Contents](#toc)

# In[ ]:


get_ipython().system('apt install zip')


# In[ ]:


cd /kaggle/train_images


# In[ ]:


get_ipython().system('zip -r -m -1 -q /kaggle/working/train_images.zip *')


# In[ ]:


cd /kaggle/test_images


# In[ ]:


get_ipython().system('zip -r -m -1 -q /kaggle/working/test_images.zip *')

