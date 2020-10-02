#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


from fastai import *
from fastai.vision import *
from fastai.metrics import error_rate


# In[ ]:


path = Path('../input/kvasir-dataset-v2/kvasir-dataset-v2')
path.ls()


# In[ ]:


bs =64

tfms = get_transforms(do_flip=True)


# In[ ]:


data= ImageDataBunch.from_folder(path,ds_tfms=tfms,bs=bs,valid_pct=0.2,no_check=True,size=128)


# In[ ]:


data.classes


# In[ ]:


data.show_batch(rows=3 , figsize=(8,8))


# In[ ]:


learn = create_cnn(data,models.resnet34,metrics=error_rate,model_dir='/tmp/models')


# In[ ]:


lr_find(learn)
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.save('Kvasir-stage1')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
losses,idx = interp.top_losses()
len(data.valid_ds)== len(losses)==len(idx)


# In[ ]:


interp.plot_top_losses(9,figsize=(6,6))


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12),dpi=60)


# In[ ]:


interp.most_confused(min_val=2)


# In[ ]:


learn.load('Kvasir-stage1')


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(4,max_lr=slice(1e-6,1e-4))


# In[ ]:


learn.save('Kvest-stage-2')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()
len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))

