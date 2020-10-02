#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai import *
from fastai.vision import *


# In[2]:


bs = 128


# In[3]:


path = "../input/nonsegmentedv2/"


# In[4]:


data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(do_flip=False), 
                                  size=224, num_workers=0, 
                                  bs=bs, valid_pct=0.2).normalize(imagenet_stats)


# In[5]:


data.show_batch(rows=3, figsize=(7,6))


# In[6]:


learn = cnn_learner(data, models.resnet18, metrics=accuracy, model_dir="/tmp/model/")


# In[7]:


learn.fit_one_cycle(5)


# In[8]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[9]:


interp.plot_top_losses(9, figsize=(15,11))


# In[10]:


interp.plot_confusion_matrix(figsize=(5,5), dpi=120)


# In[11]:


learn.save("/kaggle/working/non-wd-stage-1")


# In[12]:


learn.unfreeze()


# In[13]:


learn.lr_find()


# In[14]:


learn.recorder.plot()


# In[15]:


learn.fit_one_cycle(5, max_lr=slice(1e-6,1e-4))


# In[16]:


learn.save("/kaggle/working/non-wd-stage-2")


# In[17]:


learn.unfreeze()


# In[18]:


learn.lr_find()
learn.recorder.plot()


# In[19]:


learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))


# In[20]:


learn.save("/kaggle/working/non-wd-stage-3")


# ## Weight decay = 0.1
# 
# The default weight decay in fastai is 1e-2 which is a little conservative. We try 1e-1 and see if we can get better results.

# In[21]:


learn = cnn_learner(data, models.resnet18, metrics=accuracy, model_dir="/tmp/model/", wd=1e-1)


# In[22]:


learn.fit_one_cycle(5)


# In[23]:


learn.save("/kaggle/working/wd-stage-1")


# In[24]:


learn.unfreeze()


# In[25]:


learn.lr_find()
learn.recorder.plot()


# In[26]:


learn.fit_one_cycle(5, max_lr=slice(1e-6,1e-4))


# ## Weight decay = 1

# In[27]:


learn = cnn_learner(data, models.resnet18, metrics=accuracy, model_dir="/tmp/model/", wd=10)


# In[28]:


learn.fit_one_cycle(5)


# In[29]:


learn.save("/kaggle/working/large-wd-stage-1")


# In[30]:


learn.unfreeze()


# In[31]:


learn.lr_find()
learn.recorder.plot()


# In[32]:


learn.fit_one_cycle(5, max_lr=slice(1e-6,1e-4))


# In[ ]:


learn.save("/kaggle/working/large-wd-stage-2")


# In[ ]:




