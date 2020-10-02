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
# path = "../input/"
get_ipython().run_line_magic('matplotlib', 'inline')

# Any results you write to the current directory are saved as output
get_ipython().system('pip install fastai -U')
# !pip install fastai==1.0.46 --force-reinstall


# In[ ]:


get_ipython().system('pip list')


# In[ ]:


import torch
import fastai

from fastai import *
from fastai.vision import *

print(torch.__version__)
print(fastai.__version__)

print(torch.cuda.is_available())
print(torch.backends.cudnn.enabled)


# In[ ]:


get_ipython().system('pip install torch -U')


# In[ ]:


get_ipython().system('pip install torchvision -U')


# In[ ]:


path = untar_data(URLs.DOGS)
path


# In[ ]:


data = ImageDataBunch.from_folder(path, bs=16, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
data.show_batch(rows=3)


# In[ ]:


len(data.train_ds)


# In[ ]:


len(data.valid_ds)


# In[ ]:


learner = create_cnn(data, models.resnet34, metrics=accuracy).to_fp16()


# In[ ]:


learner.fit_one_cycle(1)


# In[ ]:


learner.unfreeze()
learner.fit_one_cycle(1, slice(1e-5,3e-4), pct_start=0.05)


# Need to switch back to fp32 for TTA because torch.stack doesn't yet work with FP16. GitHub issue [here.](https://github.com/fastai/fastai/issues/1203)

# In[ ]:


learner.to_fp32()


# In[ ]:


accuracy(*learner.TTA())


# In[ ]:


preds, y, losses = learner.get_preds(with_loss=True)


# In[ ]:


interp = ClassificationInterpretation(learner, preds, y, losses)
interp.most_confused()


# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:


interp.plot_top_losses(9, figsize=(7,7))


# In[ ]:




