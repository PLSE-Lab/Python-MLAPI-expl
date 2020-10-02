#!/usr/bin/env python
# coding: utf-8

# Hi, This is my first kernel . I've been doing the fastai course and found this dataset. So I decided to practice with a bit.

# In[ ]:


cp -r /kaggle/input/images /tmp


# In[ ]:


mv /tmp/images/images/validation/ /tmp/images/images/valid


# Importing my packages

# In[ ]:


import numpy as np
import pandas as pd

from fastai.vision import *
from fastai.widgets import *


# In[ ]:


path='/tmp/images/images'


# Creating a databunch from folder. Fortunately this dataset follows imagenet style, allowing me to just use a factory method.
# I'm taking the pictures with size 300X300 (the larger the better, right?)

# In[ ]:


data=ImageDataBunch.from_folder(path,ds_tfms=get_transforms(),size=300)


# Let's see our data now...

# In[ ]:


data.show_batch(rows=3,figsize=(6,6))


# I'm useing a resnet50 model and error_rate as my metric.
# The error rate is quite large. Nearly 40% error is not very impressive, given that even the model is usually very accurate.

# In[ ]:


learner=cnn_learner(data,models.resnet50,metrics=error_rate)
learner.fit_one_cycle(4)


# So I decided to fine tune the learning rate for my model with lr_find(). Might help minimizing the error rate. 

# In[ ]:


learner.lr_find()


# It seems my model performs worse after the learning rate goes above 1e-3. Also we can see a consistance downward slope of loss from 1e-5 to 1e-3. This should be our learning rate range.

# In[ ]:


learner.recorder.plot()


# So I unfreeze the model to train all of my layers.

# In[ ]:


learner.unfreeze()


# Using the optimal learning rate and fitting again.

# In[ ]:


learner.fit_one_cycle(4,max_lr=slice(1e-5,1e-3))


# This error_rate is better than before, but the improvement is not significant. At this point I think it's safe to assume that the dataset might have some bad labels and noise. 

# In[ ]:


interp=ClassificationInterpretation.from_learner(learner)


# In[ ]:


interp.plot_top_losses(9,figsize=(10,10))


# It is pretty obvious some instances have been missclassified in the dataset. 

# In[ ]:


interp.most_confused(min_val=10)


# It would be better to fix some of these labels myself. From both training and testing set.

# In[ ]:


ds, idxs = DatasetFormatter().from_toplosses(learner, ds_type=DatasetType.Valid)
ImageCleaner(ds,idxs,path)


# In[ ]:


ds, idxs = DatasetFormatter().from_toplosses(learner, ds_type=DatasetType.Train)
ImageCleaner(ds,idxs,path)


# After cleaning up as much as we can, let's fit the model again.

# In[ ]:


learner.fit_one_cycle(4,max_lr=slice(1e-5,1e-3))


# As we can see, error_rate was reduced 10%.
# I guess this is the best we can expect from a noisy dataset. 
# We could spend some more time with cleaning the Training set, improving performance significantly.

# In[ ]:




