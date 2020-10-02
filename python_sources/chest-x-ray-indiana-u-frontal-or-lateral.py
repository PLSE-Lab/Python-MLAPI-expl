#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().system('pip install "torch==1.4" "torchvision==0.5.0"')


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage

from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *


# In[ ]:


df_all = pd.read_csv('../input/chest-xrays-indiana-university/indiana_projections.csv')
df_all.info()
df_all.head()


# In[ ]:


df = df_all[['filename', 'projection']]
df.head()


# In[ ]:


path_img = Path('../input/chest-xrays-indiana-university/images/images_normalized')
path = Path('../working')
#path_img.ls()


# In[ ]:


learn = None
gc.collect()


# In[ ]:


bs = 256
tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_df(path_img, df, ds_tfms=tfms, size=256, bs=bs).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(7,6))


# In[ ]:


np.random.seed(101)
learn = cnn_learner(data, models.resnet18, metrics=accuracy)


# In[ ]:


#learn.lr_find()
#learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(3, max_lr=3e-3)


# In[ ]:


#learn.unfreeze()


# In[ ]:


#learn.fit_one_cycle(3, max_lr=slice(3e-5,3e-4))


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# In[ ]:




