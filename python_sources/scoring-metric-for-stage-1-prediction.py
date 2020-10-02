#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/data-science-bowl-2018"))

# Any results you write to the current directory are saved as output.


# This is my first CV competition on Kaggle and I want to figure out how to implement the scoring matrix for LB. Since stage 1 solution is out, I can use my stage 1 submission to verify this implementation.<br>
# I used a U-Net with dropout = 0.3, batchnorm for encoder layers. I added a few extra layers at the end of decoder output (64 filters -> 32 -> 16 -> 8 -> output) because it seemed to improve the performance by 0.02~0.04 on LB.<br>
# My best LB = 0.377 for U-Net output with cutoff threshold at 0.55. This LB score will be the baseline to check whether my implementation is correct. <br>
# I developed my model based loosely on the [U-Net starter kernel](https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277?scriptVersionId=2164855). I based my scoring metric implementation on [this kernel](https://www.kaggle.com/aglotero/another-iou-metric/notebook). Stephen's [kernel](https://www.kaggle.com/stkbailey/step-by-step-explanation-of-scoring-metric) also gave me a lot of insight on the scoring metric. Thanks to the authors of these kernel, who really helped me a lot for my first image competition ever.

# In[18]:


import os
import sys
import random
import warnings
import h5py

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

train_id = pd.read_csv('../input/stage-1-id/train_id.csv').values.tolist()
test_id = pd.read_csv('../input/stage-1-id/test_id.csv').values.tolist()

#rle decoder taken from the discussion forum
def rle_decode(rle_str, mask_shape, mask_dtype):
    s = rle_str.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(np.prod(mask_shape), dtype=mask_dtype)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask.reshape(mask_shape[::-1]).T


# I added my submission as private dataset. I hope that uploading submission which already has a published solution does not violate any kind of Kaggle honor code. Please let me know if I am in trouble.

# In[30]:


sol = pd.read_csv('../input/stage-1-solution/stage1_solution.csv')
sol_masks = []
size = []
for id in test_id:
    dummy = sol[sol['ImageId'] == id[0]]
    h, w = dummy[['Height', 'Width']].values[0]
    size.append((h, w))
    masks = np.zeros((len(dummy.index), h, w))
    for n, info in enumerate(dummy[['EncodedPixels', 'Height', 'Width']].values):
        mask_rle = rle_decode(info[0], (info[1], info[2]), np.uint8)
        masks[n] = mask_rle
    masks = np.max(masks, axis = 0)
    sol_masks.append(masks)

# Load my prediction mask
h5f = h5py.File('../input/stage-1-submission/stage_1_pred.h5', 'r')
y_pred_ = h5f['pred'][:]
h5f.close()


# Let's visualize some predicted mask vs ground truth mask.

# In[35]:


y_pred = []
for n, img in enumerate(sol_masks):
    y_pred.append((resize(np.squeeze(y_pred_[n]), sol_masks[n].shape, mode = 'constant', preserve_range = True)))
    
ix = np.random.randint(0,len(y_pred_))
f, ax = plt.subplots(1,2)
ax[0].imshow(np.squeeze(y_pred[ix]))
ax[1].imshow(np.squeeze(sol_masks[ix]))
ax[0].axis('off')
ax[1].axis('off')
ax[0].set_title('predicted mask')
ax[1].set_title('ground truth')
plt.show()


# Now let's define mIOU implementation as described by [this kernel](https://www.kaggle.com/aglotero/another-iou-metric/notebook).

# In[36]:


def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = label(y_true_in > 0.5)
    y_pred = label(y_pred_in > 0.55)
    
    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = len(y_true_in)
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
#     return np.array(np.mean(metric), dtype=np.float32)
    return metric


# In[38]:


mIOU = np.array(iou_metric_batch(sol_masks, y_pred))
print('The mean IOU is {}'.format(np.mean(mIOU)))
lowconf = np.where(mIOU<0.3)[0]


# Here we can check to see which prediction masks perform poorly. Let's use 0.3 as threshold and look at some random low confidence prediction. <br>These are worthy looking at if you want to implement some specialist. I noticed that color images are much harder to predict, and the predictor often misclassify tissues as a large block of nuclei. There are definately room for improvement.

# In[44]:


ix = np.random.choice(lowconf, 3)
f, ax = plt.subplots(3, 2, figsize = (10, 20))
for n, i in enumerate(ix):
    ax[n, 0].imshow(y_pred[i])
    ax[n, 1].imshow(sol_masks[i])
    ax[n, 0].axis('off')
    ax[n, 1].axis('off')
    ax[n, 0].set_title('Prediction')
    ax[n, 1].set_title('Mask')
    


# But my baseline score is 0.377, lower than the above mIOU (0.49). The problem with this implementation is that it uses skimage's label() method to identify how many individual masks there are. Since many human labels overlap with each other, this could cause the label() method to malfunction. <br>
# Let's quickly fix it. First let's examine the number of masks we got from using label(). Compare that with the number of masks from solution

# In[50]:


sol_masks_ct = []
for id in test_id:
    dummy = sol[sol['ImageId'] == id[0]]
    sol_masks_ct.append(len(dummy.index))
    
def mask_ct(y_true_in):
    ct = []
    for mask in y_true_in:
        labels = label(mask > 0.5)
        ct.append(len(np.unique(labels)))
    return ct

sol_cv_ct = mask_ct(sol_masks)
mask_ct_diff = np.where((np.array(sol_masks_ct) == np.array(sol_cv_ct)) == False)
print('Number of mask count discrepancy = {}.'.format(mask_ct_diff[0].shape[0]))
print('Difference between true mask count and false mask count: {}'.format(np.array(sol_masks_ct) - np.array(sol_cv_ct)))

plt.rcParams['figure.figsize'] = [10, 10]
ix = np.random.choice(mask_ct_diff[0], 3)
f, ax = plt.subplots(1,3)
for n, i in enumerate(ix):
    print('Image {}: True count = {}, false count = {}'.format(i, sol_masks_ct[i], sol_cv_ct[i]))
    ax[n].imshow(sol_masks[i])
    ax[n].axis('off')
    ax[n].set_title('Image {}'.format(i))
plt.show()


# In[57]:


#Rewriting the mIOU function to account for correct number of ground truth mask
def iou_metric_new(sol_df, sol_masks, batch, y_pred_in, print_table=False):
    labels = label(sol_masks > 0)
    y_pred = label(y_pred_in > 0.55)
    
    true_objects = len(sol_df[sol_df['ImageId'] == test_id[batch][0]].index)
    pred_objects = len(np.unique(y_pred))

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(sol_df, sol_masks, y_pred_in):
    batch_size = len(sol_masks)
    metric = []
    for batch in range(batch_size):
        value = iou_metric_new(sol_df, sol_masks[batch], batch, y_pred_in[batch])
        metric.append(value)
#     return np.array(np.mean(metric), dtype=np.float32)
    return metric


# In[58]:


mIOU_new = iou_metric_batch(sol, sol_masks, y_pred)
print('The mean IoU score for LB = {}'.format(np.mean(mIOU_new)))


# Great! This should approximate the LB score from stage 1. Notice that the mIOU is 0.365, about 0.01 below the LB score. I think this has to do with an [implementation detail mentioned by Allen Goodman](https://www.kaggle.com/c/data-science-bowl-2018/discussion/47756#270559) ("...we match the prediction object to the ground truth object by using the greatest IOU score").<br> I have not figure out this part yet but I am happy with the metric so far. Since the deadline is approaching I am going to skip this implementation detail and use the method above to measure /fine-tune my submission.
# This is my first kernel on Kaggle so please let me know if there is anything wrong. Thank you.

# In[ ]:




