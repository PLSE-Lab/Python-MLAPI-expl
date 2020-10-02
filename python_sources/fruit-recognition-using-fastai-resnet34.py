#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
from fastai import *
from fastai.vision import *
np.random.seed(42)


# In[ ]:


path = URLs.LOCAL_PATH/'../input/fruits-360_dataset/fruits-360'
path.ls()


# **Using the testing set as validation**

# In[ ]:


tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_folder((path), train="Training", valid="Test", ds_tfms=tfms, size=52)


# In[ ]:


learn = create_cnn(data, models.resnet50, metrics=error_rate, model_dir="/tmp/model/")


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(8, max_lr=slice(1e-2, 1e-1))

