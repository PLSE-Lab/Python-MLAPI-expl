#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
from pathlib import Path

from fastai import *
from fastai.vision import *

import os


# In[2]:


data_path = Path("../input/cell_images/cell_images/")


# In[3]:


data_path


# Do some data augmentation

# In[4]:


transforms = get_transforms(do_flip = True, 
                            flip_vert = True, 
                            max_rotate = 10.0, 
                            max_zoom = 1.1, 
                            max_lighting = 0.2, 
                            max_warp = 0.2, 
                            p_affine = 0.75, 
                            p_lighting = 0.75)


# Get data from folder and apply transformations to augment and normalize the data.
# Split 20% of the data to validation.

# In[5]:


data = ImageDataBunch.from_folder(data_path,
                                  train = '.',
                                  valid_pct = 0.2,
                                  size = 224,
                                  bs = 16,
                                  ds_tfms = transforms
                                 ).normalize(imagenet_stats)


# In[6]:


data.classes


# Vizualize some of the images

# In[7]:


data.show_batch(rows = 4, figsize = (7, 7))


# * Create a Learner Object with data, DenseNet model and the metrics

# In[8]:


learn = cnn_learner(data, models.densenet161 , metrics = [accuracy, error_rate], model_dir = '/tmp/model/')


# Find a good value for the learning rate

# In[ ]:


learn.lr_find()

learn.recorder.plot(suggestion = True)


# Train the model, at this moment the model is frozen

# In[ ]:


min_grad_lr = learn.recorder.min_grad_lr

learn.fit_one_cycle(20, min_grad_lr)


# In[ ]:


learn.save('first-phase')


# Now unfreeze the last convolutional layer and find an appropriate learning rate to train again

# In[ ]:


learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion = True)


# In[ ]:


min_grad_lr = learn.recorder.min_grad_lr

learn.fit_one_cycle(30, min_grad_lr)


# In[ ]:


learn.save('second-phase')


# Plot the loss

# In[ ]:


learn.recorder.plot_losses()


# Plot accuracy and error rate

# In[ ]:


learn.recorder.plot_metrics()


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# Vizualize some images the model predicted wrong

# In[ ]:


interp.plot_top_losses(9, figsize = (15, 10))


# Plot the confusion matrix

# In[ ]:


interp.plot_confusion_matrix()

