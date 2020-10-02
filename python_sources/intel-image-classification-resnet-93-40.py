#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai import *
from fastai.vision import *
import PIL


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
names = []
labels = []
broken_images = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        folder = (dirname.split('/')[5]).split('_')[1]
        if(folder == 'pred'):
            continue
        label = dirname.split('/')[6]
        try:
            img = PIL.Image.open(os.path.join(dirname, filename)) 
            if (not (img.size == (150, 150))):
                continue
            names.append(os.path.join(dirname[14:], filename))
            labels.append(label)
        except (IOError, SyntaxError) as e:
            print('Bad file:', os.path.join(dirname, filename))
            broken_images.append(os.path.join(dirname, filename))
        

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.DataFrame({
    "names": names,
    "labels": labels})
df.head()


# In[ ]:


data = ImageDataBunch.from_df('/kaggle/input', df)
data.normalize()


# In[ ]:


data.show_batch(rows=3, figsize=(5,5))


# In[ ]:


print(data.classes)
len(data.classes),data.c


# ## Training: resnet34
# Used a convolutional neural network backbone and a fully connected head with a single hidden layer as a classifier.
# 
# Train for 4 epochs (4 cycles through all our data).

# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate)


# In[ ]:


learn.model


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.save('stage-1')


# ## Results

# Let's see what results we have got.
# 
# We will first see which were the categories that the model most confused with one another. We will try to see if what the model predicted was reasonable or not. 

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


interp.most_confused(min_val=2)


# ## Unfreezing, fine-tuning, and learning rates

# *Unfreeze* our model and train some more.

# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-4))


# ## Training: resnet50

# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=accuracy)


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(8)


# In[ ]:


learn.save('stage-1-50')


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-4))


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


interp.most_confused(min_val=2)

