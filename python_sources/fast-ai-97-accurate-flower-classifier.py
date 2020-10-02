#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *
from fastai.metrics import error_rate


# In[ ]:


path = Path("/kaggle/input/")


# In[ ]:


path.ls()


# In[ ]:


data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(), size=224, valid_pct=0.2, num_workers=0, bs=64).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(7,6))


# ## Training

# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir="/tmp/model/")


# In[ ]:


import torch
torch.cuda.is_available()


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.save("stage-1-flower_recog")


# ## Results

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


interp.most_confused(min_val=2)


# ## Retraining

# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-3))


# In[ ]:


learn.model_dir = "/kaggle/working"
learn.save("stage-2-flower_recog")


# # Results Stage 2

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


interp.most_confused(min_val=2)

