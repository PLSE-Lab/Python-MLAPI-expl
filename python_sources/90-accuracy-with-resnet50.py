#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai import *
from fastai.vision import *
import os


# In[ ]:


os.listdir('../input/stanford-car-dataset-by-classes-folder/car_data/car_data')


# In[ ]:


base_dir = '../input/stanford-car-dataset-by-classes-folder/car_data/car_data/'


# In[ ]:


data = ImageDataBunch.from_folder(base_dir, train='train', valid='test', ds_tfms=get_transforms(), size=512, bs=30)


# In[ ]:


data.normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, fig_size=(40,40))


# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=accuracy)


# In[ ]:


learn.fit_one_cycle(10)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_top_losses(9, figsize=(30,30))


# In[ ]:


learn.model_dir='/kaggle/working/'


# In[ ]:


learn.save('stage-1')


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(10, max_lr=slice(5e-5, 5e-4))


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.save('stage-2')


# In[ ]:




