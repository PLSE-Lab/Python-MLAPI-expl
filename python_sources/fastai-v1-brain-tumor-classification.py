#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[ ]:


import numpy as np 
import pandas as pd 
import os,gc,pathlib
from sklearn.metrics import confusion_matrix
from fastai import *
from fastai.vision import *
from fastai.vision.models import *
print(os.listdir("../input"))
import torchvision.models as models


# # Make Data

# In[ ]:


DATA_DIR='../input/brain_tumor_dataset'


# In[ ]:


os.listdir(f'{DATA_DIR}')


# In[ ]:


data = ImageDataBunch.from_folder(DATA_DIR, train=".", 
                                  valid_pct=0.2,
                                  ds_tfms=get_transforms(flip_vert=True, max_warp=0),
                                  size=224,bs=8, 
                                  num_workers=0).normalize(imagenet_stats)
print(f'Classes: \n {data.classes}')


# In[ ]:


data.show_batch(rows=10, figsize=(10,5))


# # Model Build

# In[ ]:


learn = cnn_learner(data, models.resnet152, metrics=accuracy, model_dir="/tmp/model/")


# # Train

# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(10,3e-3)


# In[ ]:


learn.save('stage-1')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(10,max_lr=slice(1e-6,1e-5))


# In[ ]:


learn.save('stage-2')


# # Check Result

# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix(figsize=(8,8), dpi=60)

