#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai import *
from fastai.vision import *
import os
from os import listdir
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
path = "../input/grape/Grape/"
os.listdir(path)


# In[ ]:


path = Path(path); path


# In[ ]:


data = ImageDataBunch.from_folder(path, valid_pct = 0.2, size = 224)
data.show_batch(rows = 4)


# In[ ]:


data = data.normalize()


# In[ ]:


data.show_batch(rows=3, figsize=(15,11))


# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate)


# In[ ]:


learn.fit_one_cycle(5)


# In[ ]:


interpretation = ClassificationInterpretation.from_learner(learn)
losses, indices = interpretation.top_losses()
interpretation.plot_top_losses(4, figsize=(15,11))


# In[ ]:


interpretation.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


interpretation.most_confused(min_val=2)


# In[ ]:


learn.model_dir = '/kaggle/working'
learn.save('classification-1')


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion = True)


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(3, max_lr=1e-4)


# In[ ]:


learn.save('classifier-2')


# In[ ]:


learn.export('/kaggle/working/resnet34-grape.pkl')

