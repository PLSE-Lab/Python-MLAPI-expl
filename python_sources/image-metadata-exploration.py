#!/usr/bin/env python
# coding: utf-8

# I tried to look inside the image metadata.

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import glob
import pydicom

from tqdm import tqdm_notebook


# In[ ]:


PATH = '../input/siim-acr-pneumothorax-segmentation-data/pneumothorax/'
field_list = ['PatientAge', 'PatientSex', 'ViewPosition']
attr_list = []

for fp in tqdm_notebook(glob.glob(os.path.join(PATH,'dicom-images-train/*/*/*.dcm'))):
    dataset = pydicom.dcmread(fp)
    obj = {'fp': fp}
    for f in field_list:
        obj[f] = getattr(dataset,f)    
    attr_list.append(obj)
    
frame = pd.DataFrame(attr_list)


# How many values do these fields take?

# In[ ]:


frame.nunique()


# Let's look at the age of the patient

# In[ ]:


frame['PatientAge'] = frame['PatientAge'].astype(np.int32)
frame['PatientAge'].sort_values(ascending=False).head()


# Wow! I see a 400 year old man, isn't  it?

# In[ ]:


frame['PatientAge'][frame['PatientAge']>90] = 90


# In[ ]:


frame['PatientAge'].hist(bins=90)


# In[ ]:


frame['PatientSex'].value_counts()


# In[ ]:


frame['ViewPosition'].value_counts()


# In[ ]:


pd.DataFrame(frame.groupby(['PatientSex','ViewPosition'])['PatientAge'].count())


# In[ ]:


sns.distplot(frame['PatientAge'][frame['PatientSex']=='F'], bins=45)
sns.distplot(frame['PatientAge'][frame['PatientSex']=='M'], bins=45)


# There are approximately same distiributions

# In[ ]:


sns.distplot(frame['PatientAge'][frame['ViewPosition']=='AP'], bins=45)
sns.distplot(frame['PatientAge'][frame['ViewPosition']=='PA'], bins=45)


# Time to discover mask image fraction

# In[ ]:


train_mask = pd.read_csv(os.path.join(PATH,'train-rle.csv'))


# In[ ]:


# This fucntion based on the same name function from kernel:
# https://www.kaggle.com/abhishek/image-mask-augmentations
def rle2mask(rle, width=1024, height=1024):
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    if array.shape == (1,):
        return mask.reshape(width, height)
    starts = array[0::2]
    lengths = array[1::2]
    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 1
        current_position += lengths[index]
    return mask.reshape(width, height)


# In[ ]:


MASKS = train_mask[' EncodedPixels'].apply(rle2mask)
MASKS = np.array(MASKS)


# In[ ]:


fraction = [i.sum()/(1024**2) for i in tqdm_notebook(MASKS)]


# Attention! Size of this data different from count of image

# In[ ]:


train_mask['fraction'] = fraction


# In[ ]:


mask_stats = train_mask.groupby('ImageId')['fraction'].agg(['sum','count'])


# In[ ]:


frame.shape, mask_stats.shape


# In[ ]:


frame['ImageId'] = frame['fp'].apply(lambda x :os.path.split(x)[-1][:-4])


# In[ ]:


frame = frame.merge(mask_stats.reset_index())


# In[ ]:


frame['sum'].hist(bins=100, log=True)


# In[ ]:


frame['log_sum'] = np.log(frame['sum']+0.01)


# In[ ]:


plt.figure(figsize=(10,10))
sns.scatterplot(y='log_sum', x='PatientAge', data=frame, hue='PatientSex', alpha=0.5)
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
sns.scatterplot(y='log_sum', x='PatientAge', data=frame, hue='ViewPosition', alpha=0.5)
plt.show()


# Ok! Can we say that the **youngest** and oldest patients do not have pneumothorax?

# In[ ]:


frame['count'].value_counts()


# In[ ]:


plt.figure(figsize=(10,5))
sns.distplot(frame['PatientAge'][(frame['count']==1) & (frame['sum']>0)], bins=50)
sns.distplot(frame['PatientAge'][frame['count']==2], bins=50)
sns.distplot(frame['PatientAge'][frame['count']==3], bins=50)
plt.show()


# In[ ]:


plt.figure(figsize=(10,5))
sns.distplot(frame['sum'][(frame['count']==1) & (frame['sum']>0)], bins=50, norm_hist=True)
sns.distplot(frame['sum'][frame['count']==2], bins=50, norm_hist=True)
sns.distplot(frame['sum'][frame['count']==3], bins=50, norm_hist=True)
plt.show()


# In[ ]:


plt.figure(figsize=(10,5))
sns.distplot(frame['sum'][(frame['ViewPosition']=='AP') & (frame['sum']>0)], bins=50, norm_hist=True)
sns.distplot(frame['sum'][(frame['ViewPosition']=='PA') & (frame['sum']>0)], bins=50, norm_hist=True)
plt.show()


# Work in progress...
# Like, Share, Follow me...
