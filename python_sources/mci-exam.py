#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# settings
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai import *
from fastai.vision import *
import pandas as pd


# In[ ]:


path = Path('../input/data/data')
path.ls()


# In[ ]:


tfms = get_transforms(True,False)


# In[ ]:


data = ImageDataBunch.from_folder(path,train='train',test='test', valid_pct=0.2,
                                    ds_tfms=tfms, size=20, bs=20).normalize(imagenet_stats)


# In[ ]:


data


# In[ ]:


data.show_batch(rows=3)


# In[ ]:


model = models.resnet18


# In[ ]:


data.path = '/tmp/.torch/model'


# In[ ]:


learn = cnn_learner(data, model, metrics=accuracy,callback_fns=[ShowGraph])


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


lr = 1e-02


# In[ ]:


learn.save("stage-1")


# In[ ]:


learn.fit_one_cycle(80,slice(lr))


# In[ ]:


learn.save("state-2")


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(4,slice(lr))


# In[ ]:


accuracy(*learn.TTA())


# In[ ]:


learn.save("state-3")


# In[ ]:


learn.load("state-3")

