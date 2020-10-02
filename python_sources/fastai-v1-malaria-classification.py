#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *


# # Create a path to the images

# In[ ]:


base_dir = '../input/cell_images/cell_images/'
base_path = Path(base_dir)
base_path


# # Create a databunch to feed the data into the model

# In[ ]:


data = ImageDataBunch.from_folder(base_path,valid_pct=0.1,
                                 train='.',
                                 ds_tfms=get_transforms(max_warp=0,flip_vert=True),
                                 size=128,bs=32,
                                 num_workers=0).normalize(imagenet_stats)

print(f'Classes to classify: \n {data.classes}')
data.show_batch(rows=5,figsize=(7,7))


# # Create the learner

# In[ ]:


learner = create_cnn(data,models.resnet50,metrics=accuracy,model_dir='/tmp/model/')
learner.lr_find()
learner.recorder.plot()


# # Model Training

# In[ ]:


learner.fit_one_cycle(10,max_lr=slice(1e-4,1e-3))
learner.save('stage-1')


# # Plot the losses of the model

# In[ ]:


learner.recorder.plot_losses()


# # Plot the top losses of the model after training

# In[ ]:


inter = ClassificationInterpretation.from_learner(learner)
inter.plot_top_losses(9,figsize=(20,20))


# # Plot a confusion matrix to see how well the model performs

# In[ ]:


inter.plot_confusion_matrix(figsize=(10,10),dpi=75)


# # Unfreeze and train

# In[ ]:


learner.unfreeze()
learner.fit_one_cycle(2)


# # find the learning rate and plot the learning rate

# In[ ]:


learner.lr_find()
learner.recorder.plot()


# # Train the model

# In[ ]:


learner.fit_one_cycle(5, max_lr=slice(1e-6,1e-3))


# # Plot the losses of the train and validation sets

# In[ ]:


learner.recorder.plot_losses()


# # Plot the top losses

# In[ ]:


inter = ClassificationInterpretation.from_learner(learner)
inter.plot_top_losses(9,figsize=(20,20))


# # Plot confusion matrix

# In[ ]:


inter.plot_confusion_matrix(figsize=(10,10),dpi=75)
learner.save('malaria-fastai-V1')

