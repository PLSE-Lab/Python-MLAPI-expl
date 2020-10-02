#!/usr/bin/env python
# coding: utf-8

# # This kernel aims to observe the zero-out postprocessing for low pixel mask predicted by the model

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## RLE decode & encode for util

# In[ ]:


def rle_decode(rle_mask):
    '''
    rle_mask: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    if type(rle_mask)==float:
        return np.zeros([101,101])
    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(101*101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(101,101)
"""
used for converting the decoded image to rle mask

"""
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# In[ ]:


ls ../input


# ## Given any baseline valid csv

# In[ ]:


valid_pred=pd.read_csv('../input/valid-result/valid_1000_1.csv',index_col='id')

valid_ids=(valid_pred.index)
valid_truth=pd.read_csv('../input/tgs-salt-identification-challenge/train.csv',index_col='id')

valid_truth=valid_truth.loc[valid_ids,:]


# In[ ]:


valid_pred.head()


# In[ ]:


valid_truth.head()


# ## Firat take a look how many masks are "NAN" in general and what're the predictions look like

# In[ ]:


nan_mask=[]
pred_mask=[]
for ids in valid_truth.index:
    if (type(valid_truth.loc[ids,'rle_mask'])==float):
        pred_mask.append(valid_pred.loc[ids,'rle_mask'])
        nan_mask.append(ids)
print(len(nan_mask))
print(len(pred_mask))
    


# In[ ]:


import matplotlib.pyplot as plt
count=1
plt.figure(figsize=(30,30))
for i in range(391):
    #print(type(pred_mask[i]))
    if type(pred_mask[i])!=float:
        plt.subplot(5,5,count)
        plt.imshow(rle_decode(pred_mask[i]))
        count+=1


# ## It seems that more than half wrong masks are low pixel mask and several wrong masks have "broken" pattern around the edge while the others are general error made by the model.

# In[ ]:


def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in
    
    true_objects = 2
    pred_objects = 2

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
    return np.mean(metric)


# ## Let's see how can we improve our prediction mask by zero out low pixel masks

# In[ ]:


truth=[]
pred=[]
for i in valid_truth.index:
    truth.append(rle_decode(valid_truth.loc[i,'rle_mask']))
    pred.append(rle_decode(valid_pred.loc[i,'rle_mask']))
print('origin iou:', iou_metric_batch(truth,pred))
for pixel_thres in [5,10,20,30,40,50,60,70,80,90,100]:
    for index,img in enumerate(pred):
        pixel=np.sum(img)
        if pixel and pixel<pixel_thres:
            print(np.sum(img))
            pred[index]=np.zeros([101,101])
    print('zero out iou for mask under '+str(pixel_thres)+' pixels: '+str(iou_metric_batch(truth,pred)))


# ## Seems zeros out post-process gives 4% error reduction and 0.8% increase on IOU
# ## For further, we may use a "broken" edge detector for the rest of the wrong predictions
# # Happy Kaggling

# In[ ]:




