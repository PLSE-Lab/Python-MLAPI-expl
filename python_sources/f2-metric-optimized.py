#!/usr/bin/env python
# coding: utf-8

# # F2 metric for evaluation

# A take on optimizing the calculation of F2 score by @raresbarbantan<br>
# [https://www.kaggle.com/raresbarbantan/f2-metric](http://)

# In[ ]:


import random
from multiprocessing import cpu_count
cpu_count = cpu_count()

import numpy as np
import pandas as pd
import cv2
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

FOLDER = '../input/'

import os
print(os.listdir(FOLDER))


# In[ ]:


df = pd.read_csv('../input/train_ship_segmentations.csv')
df.head()


# # Utility functions

# RLE decoding "borrowed" from @inversion's [Run Length Decoding - Quick Start
# ](https://www.kaggle.com/inversion/run-length-decoding-quick-start )

# In[ ]:


def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


# I define some utility functions for:
# * reading an image
# * reading all masks for a given image (for training and validation purposes)
# * flattening all masks (ships) for an image into a single one (for visualisation and possible training purposes)

# In[ ]:


def read_image(img_name, type='train'):
    if type=='train':
        path = '../input/train/{}'
    else:
        path = '../input/test/{}'
    img = cv2.imread(path.format(img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def read_masks(img_name):
    mask_list = df.loc[df['ImageId'] == img_name, 'EncodedPixels'].tolist()
    all_masks = np.zeros((len(mask_list), 768,768))
    for idx, mask in enumerate(mask_list):
        if isinstance(mask, str):
            all_masks[idx] = rle_decode(mask)
    return all_masks

def read_flat_mask(img_name):
    all_masks = read_masks(img_name)
    return np.sum(all_masks, axis=0)


image_with_ships = '00021ddc3.jpg'
image_with_no_ships = '00003e153.jpg'
_, axarr = plt.subplots(1, 2, figsize=(15, 40))
axarr[0].axis('off')
axarr[1].axis('off')
axarr[0].imshow(read_image(image_with_ships))
axarr[0].imshow(read_flat_mask(image_with_ships), alpha=0.4)
axarr[1].imshow(read_image(image_with_no_ships))
axarr[1].imshow(read_flat_mask(image_with_no_ships), alpha=0.4)


# Compute IOU for 2 given binary masks. We need to avoid division by zero when there are no ships in ground truth and prediction.
# We look at values for an image against itslef, all zeros and all ones masks.

# In[ ]:


def iou(img_true, img_pred):
    i = np.sum((img_true*img_pred) >0)
    u = np.sum((img_true + img_pred) >0) + 0.0000000000000000001  # avoid division by zero
    return i/u

m = read_flat_mask(image_with_ships)
print(iou(m, m), iou(0, np.zeros((768, 768))), iou(m, np.ones((768, 768))))

m = read_flat_mask(image_with_no_ships)
print(iou(m, m), iou(m, np.zeros((768, 768))), iou(m, np.ones((768, 768))))


# # F2 score implementation, compared to original version

# In[ ]:


thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

# original version of @raresbarbantan
def f2_prev(masks_true, masks_pred):
    # a correct prediction on no ships in image would have F2 of zero (according to formula),
    # but should be rewarded as 1
    if np.sum(masks_true) == np.sum(masks_pred) == 0:
        return 1.0
    
    f2_total = 0
    for t in thresholds:
        tp,fp,fn = 0,0,0
        ious = {}
        for i,mt in enumerate(masks_true):
            found_match = False
            for j,mp in enumerate(masks_pred):
                miou = iou(mt, mp)
                ious[100*i+j] = miou # save for later
                if miou >= t:
                    found_match = True
            if not found_match:
                fn += 1
                
        for j,mp in enumerate(masks_pred):
            found_match = False
            for i, mt in enumerate(masks_true):
                miou = ious[100*i+j]
                if miou >= t:
                    found_match = True
                    break
            if found_match:
                tp += 1
            else:
                fp += 1
        f2 = (5*tp)/(5*tp + 4*fn + fp)
        f2_total += f2
    
    return f2_total/len(thresholds)

def f2(masks_true, masks_pred):
    if np.sum(masks_true) == 0:
        return float(np.sum(masks_pred) == 0)
    
    ious = []
    mp_idx_found = []
    for mt in masks_true:
        for mp_idx, mp in enumerate(masks_pred):
            if mp_idx not in mp_idx_found:
                cur_iou = iou(mt,mp)
                if cur_iou > 0.5:
                    ious.append(cur_iou)
                    mp_idx_found.append(mp_idx)
                    break
    f2_total = 0
    for th in thresholds:
        tp = sum([iou > th for iou in ious])
        fn = len(masks_true) - tp
        fp = len(masks_pred) - tp
        f2_total += (5*tp)/(5*tp + 4*fn + fp)    

    return f2_total/len(thresholds)
    
    
m = read_masks(image_with_ships)
print(f2_prev(m, m), f2_prev(m, [np.zeros((768, 768))]), f2_prev(m, [np.ones((768, 768))]))
print(f2(m, m), f2(m, [np.zeros((768, 768))]), f2(m, [np.ones((768, 768))]))

m = read_masks(image_with_no_ships)
print(f2_prev(m, m), f2_prev(m, [np.zeros((768, 768))]), f2_prev(m, [np.ones((768, 768))]))
print(f2(m, m), f2(m, [np.zeros((768, 768))]), f2(m, [np.ones((768, 768))]))


# # Let's compare speed of these versions:

# In[ ]:


sample_size = 2000
random_files = random.sample(list(df['ImageId'].unique()), sample_size)


# In[ ]:


def f2_from_fnames(fnames):
    score_sum = 0
    for fname in fnames:
        mask = read_masks(fname)
        score_sum += f2(mask, [np.zeros((768,768))])
    return score_sum

def f2_from_fnames_prev(fnames):
    score_sum = 0
    for fname in fnames:
        mask = read_masks(fname)
        score_sum += f2_prev(mask, [np.zeros((768,768))])
    return score_sum

print("Correct value is ratio of empty files: ", 
      df[df['ImageId'].isin(random_files)]['EncodedPixels'].isna().sum() / sample_size)

get_ipython().run_line_magic('time', 'scores = f2_from_fnames_prev(random_files)')
print(scores/sample_size)
get_ipython().run_line_magic('time', 'scores = f2_from_fnames(random_files)')
print(scores/sample_size)


n_workers = cpu_count
get_ipython().run_line_magic('time', 'scores = Parallel(n_workers)(delayed(f2_from_fnames_prev)(random_files[i::n_workers]) for i in range(n_workers))')
scores = sum(scores)
print(scores/sample_size)
get_ipython().run_line_magic('time', 'scores = Parallel(n_workers)(delayed(f2_from_fnames)(random_files[i::n_workers]) for i in range(n_workers))')
scores = sum(scores)
print(scores/sample_size)


# There's a substantial increase due to more efficient implementation, and another increase due to parallelizing, compared to the original version of **@raresbarbantan**.<br>
# Overall it's 5+ times faster.

# In[ ]:




