#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *
import pandas as pd
import os


# In[ ]:


image_data = Path('../input/dataset2-master/dataset2-master/images/')


# In[ ]:


image_data.ls()


# In[ ]:


np.random.seed(42)
data = ImageDataBunch.from_folder(image_data, train='TRAIN', valid='TEST', size = 128, bs=32, num_workers=0).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(12,8))


# In[ ]:


model_path=Path('/tmp/models/')
learn = cnn_learner(data, models.resnet34, metrics = error_rate, model_dir=model_path)


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, max_lr=1e-2)


# In[ ]:


learn.save("stage-1")


# In[ ]:


learn.load("stage-1")


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, max_lr = slice((1e-4)/2))


# In[ ]:


learn.save("stage-2")


# In[ ]:


learn.load("stage-2")


# In[ ]:


np.random.seed(42)
data = ImageDataBunch.from_folder(image_data, train = 'TRAIN', valid = 'TEST',size = 240, bs=16, num_workers=0).normalize(imagenet_stats)


# In[ ]:


learn.data = data
data.train_ds[0][0].shape


# In[ ]:


learn.freeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(2, max_lr=slice(1e-3, 1e-2))


# In[ ]:


learn.save("stage-3")


# In[ ]:


learn.load("stage-3")


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(2, max_lr=slice(1e-3,1e-2))


# In[ ]:


learn.save("stage-4")


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix(figsize = (8,8))


# In[ ]:




