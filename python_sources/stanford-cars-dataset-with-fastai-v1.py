#!/usr/bin/env python
# coding: utf-8

# **Stanford Cars Dataset with Fastai V1**

# Previously I have found image classification problems to be challenging when there are a large number of image classes.  Here I test the default [Fastai](http://fast.ai) image classifier against a dataset containing 16,185 images of 196 classes of cars.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *

import torch
print('pytorch version: ',torch.__version__)
import fastai
print('fastai version: ',fastai.__version__)

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().system("pip freeze > '../working/requirements.txt'")


# In[ ]:


img_dir='../input/car_data/car_data/train'
path=Path(img_dir)
data = ImageDataBunch.from_folder(path, train=".", 
                                  valid_pct=0.2,
                                  ds_tfms=get_transforms(do_flip=False,flip_vert=False, max_rotate=0,max_lighting=0.3),
                                  size=224,bs=64, 
                                  num_workers=0).normalize(imagenet_stats)
print(f'Classes: \n {data.classes}')
data.show_batch(rows=8, figsize=(40,40))


# In[ ]:


learn = create_cnn(data, models.resnet34, metrics=accuracy, model_dir="/tmp/model/")
learn.fit_one_cycle(35)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(40,40), dpi=60)


# 65% accuracy is much better than random chance (given 196 very similar object categories) and is a pretty decent result given so few lines of code.
# 
# Note that this result is slightly less accurate than the result when tested against the [Caltech256](https://www.kaggle.com/paultimothymooney/caltech-256-dataset-with-fastai-v1) and [SimpsonsCharacters](https://www.kaggle.com/paultimothymooney/simpsons-characters-dataset-with-fastai-v1) datasets.

# Credit: Adapted from https://course.fast.ai/videos/?lesson=1
