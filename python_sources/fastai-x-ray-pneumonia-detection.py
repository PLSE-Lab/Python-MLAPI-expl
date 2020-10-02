#!/usr/bin/env python
# coding: utf-8

# ## A fastai implementation of a X-Ray pneumonia detector

# Essential notebook initialization code for automatic library reload

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# Importing all necessary packages

# In[ ]:


from fastai import *
from fastai.vision import *
from fastai.imports import *

import os


# ### Quick look at the data

# We are going to use Paul Mooney's Kaggle dataset (https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

# In[ ]:


data_path = Path('../input/chest_xray/chest_xray')
data_path


# In[ ]:


np.random.seed(42)
data = ImageDataBunch.from_folder(path=data_path,
                                  train='train',
                                  valid='val',
                                  test='test',
                                  size=224,
                                  bs=64,
                                  valid_pct=0.2,
                                  ds_tfms=get_transforms())
data.normalize(imagenet_stats)
data.show_batch(rows=5)


# ### Training phase

# We are going to use rasnet50, feel free to use other models, you can find fastai supported models here docs.fast.ai/vision.models.html

# In[ ]:


learner = cnn_learner(data=data,
                      base_arch=models.resnet50,
                      metrics=error_rate,
                      model_dir="/tmp/model/")


#  Training the network using Leslie Smith's 1cycle policy

# In[ ]:


learner.fit_one_cycle(4)


# Unfreez the network so we'll train on the entire model

# In[ ]:


learner.unfreeze()


# In[ ]:


learner.save('stage-1')


# In[ ]:


learner.lr_find()
learner.recorder.plot()


# In[ ]:


learner.fit_one_cycle(10, max_lr=slice(3e-6, 3e-5))
learner.recorder.plot_losses()


# In[ ]:


interp = ClassificationInterpretation.from_learner(learner)
losses,idxs = interp.top_losses()
len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))

