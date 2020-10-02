#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
print(os.listdir('../input'))


# In[2]:


from fastai import *
from fastai.vision import *


# In[3]:


print(os.listdir('../input/rps-cv-images/'))


# In[4]:


path = Path('../input/rps-cv-images')
print(path)
print(path.ls())


# In[5]:


data = ImageDataBunch.from_folder(
    path=path,
    train=".",
    valid_pct=0.1,
    size=224,
    ds_tfms=get_transforms()
)
data.normalize(imagenet_stats)


# In[6]:


data.show_batch(rows=3, figsize=(7,6))


# In[7]:


learner = cnn_learner(data, models.resnet34, metrics=[accuracy, error_rate], model_dir='/tmp/models/')


# In[8]:


learner.lr_find()
learner.recorder.plot()


# In[9]:


lr = 1e-03
learner.fit_one_cycle(4, max_lr=slice(lr))


# In[10]:


learner.save('stage-1-frozen-resnet34')


# In[11]:


learner.recorder.plot_losses()


# In[12]:


learner.unfreeze()


# In[13]:


learner.lr_find()
learner.recorder.plot()


# In[14]:


learner.fit_one_cycle(4, max_lr=slice(1e-04))


# In[15]:


learner.save('stage-2-unfrozen-resnet34')


# In[16]:


learner.recorder.plot_losses()


# In[17]:


learner.recorder.plot_lr()


# In[18]:


interp = ClassificationInterpretation.from_learner(learner)


# In[19]:


interp.plot_top_losses(9, figsize=(15,11))


# In[20]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)

