#!/usr/bin/env python
# coding: utf-8

# ## Identifying Rose Species
# This notebook will help to build a model which will identify three species of Roses.
# 
# - rosa-centifolia
# - rosa-canina and 
# - rosa-glauca
# 
# The dataset is manually which contains 399 items. The objective is to see the model performance with relatively a small dataset. This modelling will be done using transfer learning on resnet architecture. The pretrained model is available on pytorch resnet50 model. (https://download.pytorch.org/models/resnet50-19c8e357.pth)

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
from fastai import *
from fastai.vision import *
import os
import time


# In[ ]:


path = Path('../input/rosespecies/roses')


# In[ ]:


start_time = time.time()


# In[ ]:


src = ImageList.from_folder('../input/rosespecies/roses/').split_by_rand_pct(0.2).label_from_folder()
tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
data =src.transform(tfms , size = 128).databunch(bs = 16).normalize(imagenet_stats)


# In[ ]:


data


# In[ ]:


data.show_batch(rows=3, figsize=(10,10))


# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=[error_rate], model_dir="/tmp/model/")


# In[ ]:


learn.lr_find(); learn.recorder.plot()


# ## Learning rate = 1e-4

# In[ ]:


learn.fit_one_cycle(10, 1e-4)


# Since further training has degraded the performance of the model, this is now being not done.

# In[ ]:


#learn.unfreeze(); learn.lr_find(); learn.recorder.plot()


# In[ ]:


#learn.fit_one_cycle(5, 6* 1e-5)


# ## Time to train (approximate, excluding lr find time)

# In[ ]:


time.time() - start_time


# ## Inference

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()
len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_confusion_matrix(figsize=(8,8))


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.most_confused(min_val = 1)


# In[ ]:


learn.export("/kaggle/working/roses-model.pkl")
save_texts('classes.txt', data.classes)

