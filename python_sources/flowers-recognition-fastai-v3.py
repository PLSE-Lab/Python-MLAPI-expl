#!/usr/bin/env python
# coding: utf-8

# # Flowers Recognition
# 

# In[ ]:


from fastai import *
from fastai.vision import *


# In[ ]:


import numpy as np 
import pandas as pd 

import os
print(os.listdir("../input"))
from glob import glob
import random
import cv2
import matplotlib.pylab as plt
import random as rand
import keras
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, MaxPool2D, BatchNormalization, Input
from keras.models import Sequential
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from pathlib import Path
from keras.optimizers import Adam,RMSprop,SGD


# ## View data

# In[ ]:


path =  Path('../input/flowers-recognition/flowers/flowers')


# In[ ]:


np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
                                  ds_tfms=get_transforms(), size=64, num_workers=0).normalize(imagenet_stats)


# In[ ]:


data.classes


# In[ ]:


data.show_batch(rows=3, figsize=(10,10))


# In[ ]:


learn = create_cnn(data, models.resnet34, metrics=error_rate, model_dir="/tmp/model/")


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.save('stage-1')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(2, max_lr=slice(1e-4,1e-3))


# In[ ]:


learn.save('stage-2')


# ## Interpretation

# In[ ]:


learn.load('stage-2');


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix()


# Better the model

# In[ ]:


learn.lr_find()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr = 0.01
learn.fit_one_cycle(5, slice(1e-5, lr/5))


# In[ ]:


learn.fit_one_cycle(5, max_lr=slice(1e-5,1e-4))


# In[ ]:


learn.save('stage-3')


# In[ ]:


np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=256, num_workers=0).normalize(imagenet_stats)

learn.data = data
data.train_ds[0][0].shape


# In[ ]:


learn.freeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr=1e-2/2


# In[ ]:


learn.fit_one_cycle(5, slice(lr))


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(5, slice(1e-5, lr/5))


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.save('stage-5')


# ## Interpretation 2.0

# In[ ]:


learn.load('stage-5');


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix()


# # Predictions

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))

