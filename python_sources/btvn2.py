#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# settings
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# load libraries
from fastai import *
from fastai.vision import *
import pandas as pd


# In[ ]:


size = 96 # size of input images
bs = 32 # batch size
tfms = get_transforms(do_flip=False,)


# In[ ]:


path = Path('../input/dogs-dataset-traintest/dogs-dataset-traintest/cropped'); path.ls()


# In[ ]:


# Load data to DataBunch
data = ImageDataBunch.from_folder(path,train='train',test='test',valid_pct=.2,
                                 ds_tfms=tfms, size=size, bs=bs).normalize(imagenet_stats)
data


# In[ ]:


data.show_batch(rows=3)


# ### Create your learner

# In[ ]:


model = models.resnet18


# In[ ]:


data.path = '/tmp/.torch/models'


# In[ ]:


learn = cnn_learner(data, model, metrics=accuracy,callback_fns=[ShowGraph])


# In[ ]:


learn.summary()


# ## Training begin

# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr = 10e-3


# In[ ]:


learn.fit_one_cycle(4,slice(lr))


# In[ ]:


learn.save("stage-1")


# In[ ]:


learn.unfreeze()


# In[ ]:


lr = lr /100
learn.fit_one_cycle(4,slice(lr))


# In[ ]:


accuracy(*learn.TTA())


# In[ ]:


learn.save('stage-2')


# ### Trainning stage 2

# In[ ]:


learn.load('stage-2')
pass


# In[ ]:


size = 224


# In[ ]:


# train with the change in images size
data = ImageDataBunch.from_folder(path,train='train',test='test',valid_pct=.2,
                                 ds_tfms=tfms, size=size, bs=bs).normalize(imagenet_stats)
data


# In[ ]:


learn.data = data


# In[ ]:


learn.freeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr = 1e-3


# In[ ]:


learn.fit_one_cycle(5,slice(lr))


# In[ ]:


learn.unfreeze()


# In[ ]:


lr = lr/100
learn.fit_one_cycle(5,slice(lr))


# In[ ]:


accuracy(*learn.TTA())


# In[ ]:


learn.save('stage-3')


# # Interpretation

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.most_confused(min_val=2)

