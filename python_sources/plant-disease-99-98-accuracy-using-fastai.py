#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai import *
from fastai.vision import *


# Using 224 x 224 px as input image size and slpitting data as 80:20 ratio
# 

# In[ ]:


data=ImageDataBunch.from_folder('../input/plantdisease',valid_pct=0.20,ds_tfms=get_transforms(),size=224).normalize(imagenet_stats)


# using Densenet modeland accuracy as evaluation matrics

# In[ ]:


model=cnn_learner(data,models.densenet201,metrics=accuracy)


# training model to run 7 epochs

# In[ ]:


model.fit_one_cycle(6)


# In[ ]:




