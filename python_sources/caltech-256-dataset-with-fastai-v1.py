#!/usr/bin/env python
# coding: utf-8

# **Caltech 256 Dataset with Fastai V1**
# 
# Over 30,000 images in 256 object categories
# 

# Previously I have found image classification problems to be challenging when there are a large number of image classes.  Here I test the default [Fastai](http://fast.ai) image classifier against a dataset containing 256 object categories.

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


img_dir='../input/256_objectcategories/256_ObjectCategories'
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
learn.fit_one_cycle(25)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(40,40), dpi=60)


# 85% accuracy is pretty good given 256 categories and so few lines of code! 
# 
# Note that this result is slightly more accurate than the result when tested against the [StanfordCars](https://www.kaggle.com/paultimothymooney/stanford-cars-dataset-with-fastai-v1) dataset, but is slightly less accurate than the result when tested against the [SimpsonsCharacters](https://www.kaggle.com/paultimothymooney/simpsons-characters-dataset-with-fastai-v1) dataset.

# Credit: Adapted from https://course.fast.ai/videos/?lesson=1
