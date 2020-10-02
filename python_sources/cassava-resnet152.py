#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai.vision import *
from fastai import *
from fastai.utils import *
import fastprogress
import fastai

def disable_progress():
    fastprogress.fastprogress.NO_BAR = True
    master_bar, progress_bar = fastprogress.force_console_behavior()
    fastai.basic_train.master_bar, fastai.basic_train.progress_bar = master_bar, progress_bar
    
def enable_progress():
    fastai.basic_train.master_bar, fastai.basic_train.progress_bar = fastprogress.master_bar, fastprogress.progress_bar


# In[ ]:


disable_progress()


# In[ ]:


get_ipython().system(' du -sh /kaggle/input/*')


# In[ ]:


get_ipython().system(' mkdir -p /tmp/cassava')
get_ipython().system(' cp -R ../input/train/train /tmp/cassava')


# In[ ]:


get_ipython().system(' ls /tmp/cassava/train')


# In[ ]:


path = "/tmp/cassava/train"


# In[ ]:


from fastai.vision import *


# In[ ]:


data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2, bs=12,
        ds_tfms=get_transforms(max_zoom=1.5), size=500, num_workers=4).normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet152, metrics=[error_rate, accuracy], wd=0.1)


# In[ ]:


learn.fit_one_cycle(10)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.unfreeze()
learn.lr_find(start_lr=1e-12)


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(2, max_lr=slice(1e-8, 1e-6))


# In[ ]:


learn.recorder.plot_losses()

