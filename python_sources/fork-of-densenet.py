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


from fastai.vision import *
get_ipython().system(' mkdir -p /tmp/cassava')
get_ipython().system(' cp -R ../input/train/train /tmp/cassava')
path = "/tmp/cassava/train"
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2, bs=12,
        ds_tfms=get_transforms(max_zoom=1.5), size=500, num_workers=4).normalize(imagenet_stats)
learn = cnn_learner(data, models.densenet201, metrics=[error_rate, accuracy], wd=0.1, ps=0.9)


# In[ ]:


learn.fit_one_cycle(20)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.recorder.plot_lr(show_moms=True)


# In[ ]:


learn.recorder.plot_metrics()

