#!/usr/bin/env python
# coding: utf-8

# # Building an image classifier
# Material from fast.ai lesson 1 (https://github.com/fastai/fastai/blob/master/courses/dl1/lesson1.ipynb), adapted to run on Kaggle kernels

# In[ ]:


import os
import numpy as np
import torch

from fastai.transforms import tfms_from_model
from fastai.conv_learner import ConvLearner
from fastai.model import resnet34
from fastai.dataset import ImageClassifierData
from fastai.plots import ImageModelResults


# In[ ]:


PATH = "../input/"
TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"


# In[ ]:


torch.cuda.is_available()


# In[ ]:


fnames = np.array([f'train/{f}' for f in sorted(os.listdir(f'{PATH}train'))])
labels = np.array([(0 if 'cat' in fname else 1) for fname in fnames])


# In[ ]:


labels


# In[ ]:


fnames


# In[ ]:


arch=resnet34
sz=224


# In[ ]:


learning_rate = 0.01
training_length = 2 # epochs


# In[ ]:


data = ImageClassifierData.from_names_and_array(
    path=PATH, 
    fnames=fnames, 
    y=labels, 
    classes=['dogs', 'cats'], 
    test_name='test', 
    tfms=tfms_from_model(arch, sz)
)


# In[ ]:


learner = ConvLearner.pretrained(arch, data, precompute=True, tmp_name=TMP_PATH, models_name=MODEL_PATH)
learner.fit(learning_rate, training_length)


# # Checking our model's predictions

# In[ ]:


# predictions for the validation set
log_preds = learner.predict()

# a class that will help us plot our results
results = ImageModelResults(data.val_ds, log_preds)


# In[ ]:


cats = 0
dogs = 1


# In[ ]:


results.plot_most_correct(cats)


# In[ ]:


results.plot_most_correct(dogs)


# In[ ]:


results.plot_most_incorrect(cats)


# In[ ]:


results.plot_most_incorrect(dogs)


# In[ ]:


results.plot_most_uncertain(cats)


# In[ ]:




