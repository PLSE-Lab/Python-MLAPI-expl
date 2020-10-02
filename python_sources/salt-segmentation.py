#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np 
import pandas as pd 

from fastai import *
from fastai.vision import *

import os
print(os.listdir("../input"))


# In[2]:


path=Path('../input/train')


# In[3]:


path_imgs= path/'images'
path_lbls= path/'masks'


# In[4]:


img_f= path_imgs.ls()[0]
img=open_image(img_f)
img


# In[5]:


get_labels= lambda x: path_lbls/f'{x.stem}{x.suffix}'
mask=open_mask(get_labels(img_f))
mask


# In[6]:


class SaltSegmentationLabelList(SegmentationLabelList):
    def open(self,fn): return open_mask(fn, div=True)
    
class SaltSegmentationItemList(ImageList):
    _label_cls= SaltSegmentationLabelList


# In[7]:


src=(SaltSegmentationItemList.from_folder(path/'images')
     .split_by_rand_pct()
     .label_from_func(get_labels, classes=['void','salt']))


# In[8]:


data = (src.transform(get_transforms(), size=(224,224), tfm_y=True)
        .databunch(bs=16)
        .normalize(imagenet_stats))


# In[14]:


data.show_batch(rows=3, alpha=0.5)


# In[16]:


def dice(input:Tensor, targs:Tensor, iou:bool=False)->Rank0Tensor:
    "Dice coefficient metric for binary target. If iou=True, returns iou metric, classic for segmentation problems."
    n = targs.shape[0]
    input = input.argmax(dim=1).view(n,-1)
    targs = targs.view(n,-1)
    intersect = (input*targs).sum().float()
    union = (input+targs).sum().float()
    if not iou: return 2. * intersect / union
    else: return intersect / (union-intersect+1.0)

def accuracy_salt(input, target):
    target=target.squeeze(1)
    mask =target>0
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()


# In[11]:


learner= unet_learner(data, models.resnet34, metrics=[accuracy_salt,dice], model_dir="/tmp/models/") 


# In[12]:


learner.lr_find()
learner.recorder.plot()


# In[15]:


lr= 1e-3
learner.fit_one_cycle(5,slice(lr),wd=1e-1)


# In[17]:


learner.unfreeze()


# In[18]:


learner.lr_find()
learner.recorder.plot()


# In[21]:



learner.fit_one_cycle(10,slice(1e-5,lr/3),wd=1e-1)


# In[22]:


learner.save('salt-1')


# In[ ]:


learner.to_fp16()


# In[24]:



learner.fit_one_cycle(5,slice(1e-5,lr/3),wd=1e-1)


# In[ ]:




