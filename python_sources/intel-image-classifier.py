#!/usr/bin/env python
# coding: utf-8

# Image Classifier - Intel Image Classification (using fast.ai's first lesson as reference)

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


print(os.listdir("../input/seg_train/seg_train"))


# In[ ]:


from fastai.vision import *
from fastai.metrics import error_rate


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


img_dir = "../input/"


# In[ ]:


path = Path(img_dir)
path


# In[ ]:


tfms = get_transforms(do_flip=False)
data = (ImageList.from_folder(path)
        .split_by_folder(train='seg_train', valid='seg_test')
        .label_from_folder()
        .transform(tfms, size=224)
        .databunch())  


# In[ ]:


data.classes


# In[ ]:


len(data.classes)


# In[ ]:


learn = cnn_learner(data, models.resnet34, model_dir = '/tmp/models', metrics=error_rate)


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_top_losses(9, figsize=(15, 11))


# In[ ]:


interp.most_confused(min_val=2)


# In[ ]:


learn.save('stage-1')


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-5, 1e-3))


# In[ ]:




