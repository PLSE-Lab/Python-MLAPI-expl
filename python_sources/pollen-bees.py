#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from fastai.vision import *
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


# In[ ]:


print(os.listdir("../input/pollendataset/PollenDataset"))


# In[ ]:


images = get_image_files("../input/pollendataset/PollenDataset/images")


# In[ ]:


def get_labels(f_path): return 'N' if 'N' in str(f_path) else 'P'


# In[79]:


f_path = "../input/pollendataset/PollenDataset/images"
bs = 64
tfms = get_transforms(flip_vert=False, max_zoom=1.)
data = ImageDataBunch.from_name_func(f_path, images, label_func=get_labels, ds_tfms=tfms, size=224, bs=bs
                                    ).normalize(imagenet_stats)


# In[58]:


data.show_batch(rows=3, figsize=(10,9))


# In[60]:


data.classes, data.c, len(data.train_ds), len(data.valid_ds)


# In[69]:


learn = cnn_learner(data, models.resnet50, metrics=error_rate, model_dir='/kaggle/working')


# In[62]:


learn.lr_find()
learn.recorder.plot()


# In[63]:


lr = 0.01


# In[70]:


learn.fit_one_cycle(7, slice(lr))


# In[71]:


learn.recorder.plot_losses()


# In[72]:


learn.save('stage-1-rn50')


# In[73]:


learn.unfreeze()


# In[74]:


learn.lr_find()
learn.recorder.plot()


# In[75]:


learn.fit_one_cycle(5, slice(1e-4, 0.0007))


# In[76]:


learn.recorder.plot_losses()


# In[77]:


learn.save('stage-2-rn50')


# ## Increasing size of images

# In[80]:


data2 = ImageDataBunch.from_name_func(f_path, images, label_func=get_labels, ds_tfms=tfms, size=299, bs=bs
                                    ).normalize(imagenet_stats)


# In[81]:


learn.data = data2


# In[82]:


learn.freeze()


# In[83]:


learn.lr_find()
learn.recorder.plot()


# In[85]:


lr = 1e-3/2


# In[86]:


learn.fit_one_cycle(5, slice(lr))


# In[87]:


learn.save('stage-1-299-rn50')


# In[89]:


learn.unfreeze()


# In[90]:


learn.fit_one_cycle(5, slice(1e-5, lr/5))


# In[91]:


learn.recorder.plot_losses()


# ## Interpretation

# In[92]:


interp = ClassificationInterpretation.from_learner(learn)


# In[93]:


interp.plot_confusion_matrix()


# In[96]:


interp.plot_top_losses(2, figsize=(15,11), heatmap=False)

