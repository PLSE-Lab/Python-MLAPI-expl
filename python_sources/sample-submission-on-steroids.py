#!/usr/bin/env python
# coding: utf-8

# This is a simple tweak to sample_submission.csv file to improve its score. It is not a serious work, just having fun and playing with the data. Following the spirit of [better-sample-submission kernel](https://www.kaggle.com/raddar/better-sample-submission) by raddar. 
# 
# The idea is that duplicated ids in sample_submission.csv file indicate instances with more than one mask. For our needs here we just use the knowledge that such instances have a mask. There are 78 such instances.
# 
# Next, we look at the train annotations and collect average statistic of masks locations. Our new average mask is comprised of all pixels where there are at least 20 train data masks with that pixel on. This is the mask that we assign to each of the 78 instances. Checkout the images!

# In[ ]:


import sys
import pandas as pd
import time
import matplotlib.pyplot as plt

sys.path.insert(0, '../input/siim-acr-pneumothorax-segmentation')

from mask_functions import *


# In[ ]:


df = pd.read_csv('../input/siim-acr-pneumothorax-segmentation-data/pneumothorax/train-rle.csv', index_col=0)
df.columns = ['map']

def get_totals():
    totals = np.zeros((1024,1024))
    for index,row in df.iterrows():
        if row['map'] != ' -1':
            mask = rle2mask(row['map'], 1024, 1024)
            totals += mask
    totals /= 255
    return totals

get_ipython().run_line_magic('time', 'totals = get_totals()')


# In[ ]:


fig, axes = plt.subplots(2, 2, figsize=(20, 20))

threshold = 20

axes[0][0].imshow(totals.T, cmap=plt.cm.bone)
axes[0][0].set_title('totals')
axes[0][1].imshow(np.clip(totals,0,200).T, cmap=plt.cm.bone)
axes[0][1].set_title('np.clip(totals,0,200)')
axes[1][0].imshow(np.clip(totals,0,100).T, cmap=plt.cm.bone)
axes[1][0].set_title('np.clip(totals,0,100)')
axes[1][1].imshow(totals.T > threshold, cmap=plt.cm.bone)
a = axes[1][1].set_title('totals > %d' % threshold)


# In[ ]:


super_mask_right = mask2rle(255*np.logical_and(totals > threshold, np.tile(range(1024),1024).reshape((1024,1024)).T >= 512),1024,1024)
super_mask_left = mask2rle(255*np.logical_and(totals > threshold, np.tile(range(1024),1024).reshape((1024,1024)).T < 512),1024,1024)

sub = pd.read_csv('../input/siim-acr-pneumothorax-segmentation/sample_submission.csv')

tmp = sub.groupby('ImageId')['ImageId'].count().reset_index(name='N')
tmp = tmp.loc[tmp.N > 1] #find image id's with more than 1 row -> has pneumothorax mask!
print('number of instances with duplicate ids', len(tmp))

# length 1484 to 1294:
sub = sub.drop(sub.loc[sub.ImageId.isin(tmp.ImageId),:].index)
pd_right = pd.DataFrame({'ImageId': tmp.ImageId, 'EncodedPixels': super_mask_right})
pd_left = pd.DataFrame({'ImageId': tmp.ImageId, 'EncodedPixels': super_mask_left})
# new length is 1450 after adding 2*78:
sub = pd.concat([sub, pd_right, pd_right, pd_left, pd_left], axis=0)

sub.to_csv('sample_submission_on_steroids.csv',index=None)


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(10, 10))
a = axes[0].imshow(rle2mask(super_mask_left,1024,1024).T)
a = axes[1].imshow(rle2mask(super_mask_right,1024,1024).T)


# In[ ]:




