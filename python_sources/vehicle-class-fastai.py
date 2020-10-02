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
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from fastai import * 
from fastai.vision import *


# In[ ]:


path=Path('/kaggle/input/vehicle/train/train/')
ImageList.from_folder(path)


# In[ ]:


data = (ImageList.from_folder(path)
        #Where to find the data? 
        .split_by_rand_pct(0.20,seed=44)
        .label_from_folder()
        #Data augmentation? -> use tfms with a size of 128
        .transform(get_transforms(do_flip=True,flip_vert= False,max_zoom=1.1, max_lighting=0.2, max_warp=0.2),size=128)
        .databunch(bs=64)
        .normalize(imagenet_stats))  
print(data.classes)
data.show_batch(rows=3, figsize=(5,5))


# In[ ]:


Path('/tmp/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)
get_ipython().system('cp ../input/resnet34/resnet34.pth /tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth')
learn = cnn_learner(data, models.resnet34, metrics=[accuracy],model_dir='/kaggle',pretrained=True)
learn.summary()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr = 1e-2
learn.fit_one_cycle(1, slice(lr))


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()
len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, slice(1e-4,1e-3))

