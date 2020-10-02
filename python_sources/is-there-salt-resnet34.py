#!/usr/bin/env python
# coding: utf-8

# Detecting Salt in images using FastAI

# In[ ]:


import numpy as np
import pandas as pd
import os
print(os.listdir("/kaggle/input/"))
#os.getcwd()


# In[ ]:


from fastai import *
from fastai.vision import *
from pathlib import Path
import json
torch.cuda.set_device(0)


# In[ ]:


MASKS_FN = 'train.csv'
TRAIN_DN = Path('train/images/')
TEST = Path('test/images/')

PATH = Path('/kaggle/input/')
TMP = Path('/kaggle/working/tmp')
MODEL = Path('/kaggle/working/model')
seg = pd.read_csv(PATH/MASKS_FN).set_index('id')
seg.head()

sz = 101
bs = 64
nw = 4


# In[ ]:


train_names_png = [TRAIN_DN/f for f in os.listdir(PATH/TRAIN_DN)]
train_names = list(seg.index.values)


# In[ ]:


test_names_png = [TEST/f for f in os.listdir(PATH/TEST)]  #not used


# In[ ]:


seg['isSalt'] = seg['rle_mask'].apply(lambda x : type(x)!=float).astype(int)


# In[ ]:


salt = seg['isSalt']
salt.to_csv('salt.csv')


# In[ ]:


x_names = [f'{x}.png' for x in train_names if(type(seg.loc[x]['rle_mask'])!=float)]
x_names_path = np.array([TRAIN_DN/x for x in x_names])


# In[ ]:


x_names[0:5]


# In[ ]:


tfms = get_transforms(do_flip=True, max_rotate=10, max_zoom=1.2, max_lighting=0.3, max_warp=0.15)
md = image_data_from_csv(path = '/kaggle/working/', folder=PATH/TRAIN_DN, 
                         ds_tfms = tfms, tfms = imagenet_norm, 
                         suffix= '.png', csv_labels='salt.csv')


# In[ ]:


x,y = next(iter(md.train_dl))


# In[ ]:


x.shape, y


# In[ ]:


arch = tvm.resnet34
learn = ConvLearner(md, arch, metrics=accuracy)


# # Training Resnet34

# In[ ]:


lr_find(learn)
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.unfreeze()


# In[ ]:


lr = 1E-3
lrs = np.array([lr/10,lr/3,lr])
learn.fit(lrs,2,cycle_len=10,use_clr=(20,5))


# In[ ]:


wd = 1E-4


# In[ ]:


learn.fit(lr/10,2,cycle_len=10, wds = wd, use_clr=(20,5))


# In[ ]:


learn.sched.plot_loss()


# In[ ]:


learn.save('resnet34_issalt')


# In[ ]:


predsTTA, y = learn.TTA();


# In[ ]:


log_preds = predsTTA.mean(0)


# In[ ]:


accuracy_np(log_preds,y) #Not bad


# In[ ]:


testTTA = learn.TTA(is_test=True);
probs = np.exp(testTTA[0].mean(0));
probs.shape


# In[ ]:


probs[0:5,:]


# In[ ]:


np.save('probs_issalt', probs)


# In[ ]:




