#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from fastai.vision import *
from pathlib import Path
import cv2


# In[ ]:


import os
print(os.listdir("../input/recursion-cellular-image-classification/"))


# In[ ]:


path = Path('../input/recursion-cellular-image-classification/')


# In[ ]:


train = pd.read_csv(f'{path}/train.csv')
train.head()


# In[ ]:


pixel_stats = pd.read_csv(f'{path}/pixel_stats.csv')
pixel_stats.head()


# In[ ]:


plt.style.use('ggplot')
plt.figure(figsize=(15,5))
train.experiment.value_counts().plot.barh()


# In[ ]:


plt.figure(figsize=(15,5))
train.sirna.value_counts()
train.sirna.value_counts().plot.bar()


# In[ ]:


len(train.sirna.unique())


# In[ ]:


train.isnull().sum().sort_index()


# In[ ]:


smpl=1
train['path'] = train['experiment']+'/Plate'+train['plate'].astype(str)+'/'+train['well'].astype(str)+'_s'+str(smpl)+'_w'


# In[ ]:


train.head()


# In[ ]:



img = cv2.imread(f"{path}/train/HUVEC-06/Plate1/B02_s1_w1.png")
plt.imshow(img)


# In[ ]:


gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.imshow(gray_img)


# In[ ]:


plt.figure(figsize=(10,5))
[[plt.subplot(1,6,i+1),plt.imshow(cv2.imread(f"{path}/train/{train.path[0]}{str(i+1)}.png")), plt.grid(False), plt.yticks([]),  plt.xticks([])] for i in range(6)];


# In[ ]:


plt.figure(figsize=(10,5))
[[plt.subplot(1,6,i+1),plt.imshow(cv2.cvtColor(cv2.imread(f"{path}/train/{train.path[0]}{str(i+1)}.png"),cv2.COLOR_RGB2GRAY)), plt.grid(False), plt.yticks([]),  plt.xticks([])] for i in range(6)];


# In[ ]:


source https://www.kaggle.com/tanlikesmath/rcic-fastai-starter
def opening_file(fn):
    return Image(pil2tensor(np.dstack([cv2.cvtColor(cv2.imread(f"{fn}{str(i+1)}.png"),cv2.COLOR_RGB2GRAY) for i in range(6)]), np.float32).div_(255))


# source https://www.kaggle.com/tanlikesmath/rcic-fastai-starter

# In[ ]:


class MultiChannelImageList(ImageList):
     def open(self, fn):
        return opening_file(fn)


# In[ ]:


image_list_df = train.copy()

image_list_df.drop(['id_code', 'experiment', 'plate', 'well'], axis=1, inplace = True)

image_list_df.head()


# In[ ]:


dat = MultiChannelImageList.from_df(df=image_list_df, path=r'train/',cols='path')


# 

# In[ ]:


#copied from source https://www.kaggle.com/tanlikesmath/rcic-fastai-starter
def image2np(image:Tensor)->np.ndarray:
    "Convert from torch style `image` to numpy/matplotlib style."
    res = image.cpu().permute(1,2,0).numpy()
    if res.shape[2]==1:
        return res[...,0]  
    elif res.shape[2]>3:
        return res[...,:3]
    else:
        return res

vision.image.image2np = image2np


# In[ ]:


data = (MultiChannelImageList.from_df(df=image_list_df,path=f'{path}/train/', cols = 'path')
        .split_by_rand_pct(0.1)
        .label_from_df(cols ='sirna')
        .transform(get_transforms(),size=128)
        .databunch(bs=8,num_workers=0)
        .normalize(imagenet_stats)
       )


# In[ ]:


arch = models.resnet34


# In[ ]:


data


# In[ ]:




