#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np 
import pandas as pd
import cv2
import skimage.io
from skimage.transform import resize, rescale

from multiprocessing import Pool

from tqdm.notebook import tqdm
import matplotlib.pyplot as plot


# In[ ]:


# https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/145618
def crop_white(image: np.ndarray) -> np.ndarray:
    assert image.shape[2] == 3
    assert image.dtype == np.uint8
    ys, = (image.min((1, 2)) != 255).nonzero()
    xs, = (image.min(0).min(1) != 255).nonzero()
    if len(xs) == 0 or len(ys) == 0:
        return image
    return image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


# In[ ]:


def crop_white_with_mask(image: np.ndarray, mask: np.ndarray) -> (np.ndarray, np.ndarray):
    assert image.shape[2] == 3
    assert image.dtype == np.uint8
    ys, = (image.min((1, 2)) != 255).nonzero()
    xs, = (image.min(0).min(1) != 255).nonzero()
    if len(xs) == 0 or len(ys) == 0:
        return image, mask
    return image[ys.min():ys.max() + 1, xs.min():xs.max() + 1], mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


# ## Load dataframe

# In[ ]:


train_labels = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv')


# In[ ]:


train_labels.head()


# In[ ]:


data_dir = '/kaggle/input/prostate-cancer-grade-assessment/train_images/'
mask_dir = '/kaggle/input/prostate-cancer-grade-assessment/train_label_masks/'


# ## Start here

# In[ ]:


save_dir = "/kaggle/train_images/"
save_mask_dir = '/kaggle/train_label_masks/'
os.makedirs(save_dir, exist_ok=True)
os.makedirs(save_mask_dir, exist_ok=True)


# In[ ]:


#for img_id in train_labels.image_id:
def crop_white_save(img_id):
    load_path = data_dir + img_id + '.tiff'
    save_path = save_dir + img_id + '.png'
    load_path_mask = mask_dir + img_id + '_mask.tiff'
    save_path_mask = save_mask_dir + img_id + '_mask.png'
    
    biopsy = skimage.io.MultiImage(load_path)
    if os.path.exists(load_path_mask):
        biopsy_mask = skimage.io.MultiImage(load_path_mask)
        # out = cv2.resize(biopsy[-1], (biopsy[-1].shape[1] // 4, biopsy[-1].shape[0] // 4))
        out, mask_out = crop_white_with_mask(biopsy[-1],biopsy_mask[-1])
        cv2.imwrite(save_path, out)
        cv2.imwrite(save_path_mask, mask_out)
        return 1
    else:
        out = crop_white(biopsy[-1])
        cv2.imwrite(save_path, out)
        return 0


# In[ ]:


with Pool(processes=4) as pool:
    has_mask = list(
        tqdm(pool.imap(crop_white_save, list(train_labels.image_id)), total = len(train_labels.image_id))
    )
print('%d / %d has mask.'%(sum(has_mask),len(has_mask)))


# In[ ]:


get_ipython().system('tar -czf train_images.tar.gz ../train_images/*.png')
get_ipython().system('tar -czf train_label_masks.tar.gz ../train_label_masks/*.png')

