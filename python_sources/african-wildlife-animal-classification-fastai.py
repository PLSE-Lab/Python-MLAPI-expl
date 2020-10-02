#!/usr/bin/env python
# coding: utf-8

# # This notebook classifies 4 african wildlife using the fastai library.
# 
# **Buffalos | Elephants | Rhino | Zebra**

# The fastai library is based on research into deep learning best practices undertaken at fast.ai, and includes "out of the box" support for vision, text, tabular, and collab (collaborative filtering) models. This library allows high acurracy to be achieved with little training data. In this notebook we will be using the computer vision library from fastai to train on small data set. 
# 
# The fastai library enables a friendly entry into Machine Learning for people coming from different domains to help apply ML to the fields with ease. 

# # Acknowledgements
# 
# Fastai contributers

# **Setup**

# In[ ]:


# first we import the libraries

from fastai.vision import * 
from fastai import *

import warnings
warnings.filterwarnings('ignore')


# **Load data**

# In[ ]:


path = "../input/african-wildlife" # path to data set in kaggle

np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
#imagedatabunch wraps the dataset and transforms is into the require format for training our model


# In[ ]:


#list classes
data.classes


# **Data Visualisation**

# In[ ]:


data.show_batch(rows=4, figsize=(7,8))


# **Define the model**

# In[ ]:


#the model uses tranfer learning using the resnet34
learn = cnn_learner(data, models.resnet34, metrics=accuracy, callback_fns=ShowGraph)


# **Train the model**

# In[ ]:


learn.fit_one_cycle(5) # we set the model to run for 5 epochs


# * We get an accuracy of** 97.00%** with just few training and under a short period of time and a few lines of code

# # we can also plot a confusion matrix to analyse how are model performed

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# In[ ]:




