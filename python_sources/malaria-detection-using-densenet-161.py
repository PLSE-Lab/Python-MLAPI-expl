#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from fastai.vision import *
import numpy as np


# In[3]:


image_data = Path("../input/cell_images/cell_images")


# In[4]:


image_data.ls()


# In[5]:


np.random.seed(42)
data = ImageDataBunch.from_folder(image_data, train='.', valid_pct=0.2, 
                                  ds_tfms=get_transforms(flip_vert=True, max_warp=0),size=128, bs=64,
                                  num_workers=0).normalize(imagenet_stats)


# In[6]:


data.classes, data.c


# In[7]:


data.train_ds[0][0].shape


# In[8]:


data.show_batch(rows=3)


# In[9]:


learn = cnn_learner(data, models.densenet161, metrics=accuracy, path='./')


# In[10]:


learn.lr_find()


# In[11]:


learn.recorder.plot()


# In[12]:


learn.fit_one_cycle(6, max_lr=slice(1e-04,1e-3))


# In[13]:


learn.save("stage-1")


# In[14]:


learn.unfreeze()


# In[15]:


learn.lr_find()


# In[16]:


learn.recorder.plot()


# In[17]:


learn.fit_one_cycle(6, max_lr=slice(1e-06,1e-05))


# In[18]:


learn.save("stage-2")


# In[19]:


interp = ClassificationInterpretation.from_learner(learn)
losses, idxs = interp.top_losses()
len(data.valid_ds)==len(losses)==len(idxs)


# In[20]:


interp.plot_top_losses(9, figsize=(12,12))


# In[21]:


interp.plot_confusion_matrix(figsize=(6,6))


# In[ ]:




