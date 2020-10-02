#!/usr/bin/env python
# coding: utf-8

# <center><img src="https://i.imgur.com/2rGzuN5.jpg" width="750px" style="border:1px solid black"></center>

# In this kernel, I will resize all the train images to (512, 512) because I/O with this dataset is not great.

# ## Import necessary libraries

# In[ ]:


import os
import gc
import cv2

import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from tqdm.notebook import tqdm, trange


# In[ ]:


N = 16
H = 512
W = 512
C = cv2.COLOR_BGR2RGB


# ## Define paths and load .csv files

# In[ ]:


TEST_IMG_PATH = '../input/siim-isic-melanoma-classification/jpeg/test/'
TRAIN_IMG_PATH = '../input/siim-isic-melanoma-classification/jpeg/train/'


# In[ ]:


test_df = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
train_df = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')


# ## Resize images to (512, 512) and save (with multi-threading)

# In[ ]:


def save_to_zip(out_path):
    # ZIP & rm -rf (remove) folder

    if 'test' in out_path:
        get_ipython().system('zip -r test.zip test')
        get_ipython().system('rm -rf test')

    if 'train_1' in out_path:
        get_ipython().system('zip -r train_1.zip train_1')
        get_ipython().system('rm -rf train_1')

    if 'train_2' in out_path:
        get_ipython().system('zip -r train_2.zip train_2')
        get_ipython().system('rm -rf train_2')


# In[ ]:


def save(ids, in_path, out_path):
    # Resize images to (512, 512) and save

    ids = tqdm(ids)

    for idx, image_name in enumerate(ids):
        input_read_path = in_path + image_name
        image = cv2.imread(input_read_path + '.jpg')
        image = cv2.resize(cv2.cvtColor(image, C), (H, W))
        output_write_path = out_path + image_name + '.jpg'
        cv2.imwrite(output_write_path, image); del image; gc.collect()


# In[ ]:


# Create directories and image ID lists

get_ipython().system('mkdir test')
get_ipython().system('mkdir train_1')
get_ipython().system('mkdir train_2')

length = int(0.5*len(train_df))
test_ids = np.array_split(np.array(test_df.image_name), N)
train_ids_1 = np.array_split(np.array(train_df.image_name[:length]), N)
train_ids_2 = np.array_split(np.array(train_df.image_name[length:]), N)


# In[ ]:


# Save resized test images to folders with multi-threading

path = "test/"
parallel = Parallel(n_jobs=N, backend="threading")
parallel(delayed(save)(ids, TEST_IMG_PATH, path) for ids in test_ids)

# Save resized train images to folders files with multi-threading

path = "train_1/"
parallel = Parallel(n_jobs=N, backend="threading")
parallel(delayed(save)(ids, TRAIN_IMG_PATH, path) for ids in train_ids_1)

path = "train_2/"
parallel = Parallel(n_jobs=N, backend="threading")
parallel(delayed(save)(ids, TRAIN_IMG_PATH, path) for ids in train_ids_2)


# In[ ]:


# Save test images to ZIP

save_to_zip("test")

# Save train images to ZIP

save_to_zip("train_1")
save_to_zip("train_2")


# ## Thank you :D
