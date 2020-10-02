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


tfms = get_transforms(True,False)


# In[ ]:


path = Path('../input/catsdogs/')
path.ls()


# In[ ]:


data =  ImageDataBunch.from_folder(path,train='training_set',test='test_set.zip', valid_pct=0.2,
                                    ds_tfms=tfms, size=200, bs=20).normalize(imagenet_stats)


# In[ ]:


data


# In[ ]:


data.show_batch(rows=3)


# In[ ]:


model = models.resnet18


# In[ ]:


data.path = '/tmp/.torch/dogsandcats'


# In[ ]:


learn = cnn_learner(data, model, metrics=accuracy,callback_fns=[ShowGraph])


# In[ ]:


learn.summary()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


lr = 1e-02


# In[ ]:


learn.fit_one_cycle(4,slice(lr))


# In[ ]:


learn.save("stage-1")


# In[ ]:


learn.fit_one_cycle(1,slice(lr))


# In[ ]:


learn.load("stage-1")


# In[ ]:


lr = 1e-03


# In[ ]:


learn.fit_one_cycle(2,slice(lr))


# In[ ]:


learn.save("stage-2")


# In[ ]:


learn.unfreeze()


# In[ ]:


lr = lr /100
learn.fit_one_cycle(4,slice(lr))


# In[ ]:


learn.save("stage-2")


# In[ ]:


learn.unfreeze()


# In[ ]:


lr = 1e-03
learn.fit_one_cycle(4,slice(lr))


# In[ ]:


accuracy(*learn.TTA())


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:





# In[ ]:




