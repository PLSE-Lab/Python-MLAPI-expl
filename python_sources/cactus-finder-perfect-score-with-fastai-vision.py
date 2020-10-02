#!/usr/bin/env python
# coding: utf-8

# ## Cactus Classifier with fastai.vision
# 
# This dataset is perfect for beginners, because the problem is literally trying to find stick-like patterns.Also fastai.vision library gives score 1 with little to no effort.
# I've tried densenet and resnet here and they both give perfect score. 

# In[ ]:


import numpy as np
import pandas as pd
from pathlib import Path
from fastai import *
from fastai.vision import *
import torch


# In[ ]:


data_folder = Path("../input")
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/sample_submission.csv")


# In[ ]:


test_img = ImageList.from_df(test_df, path=data_folder/'test', folder='test')


# In[ ]:


src = (ImageList.from_df(train_df, path=data_folder/'train', folder='train')
        .split_by_rand_pct(0.01)
        .label_from_df()
        .add_test(test_img)
       )


# In[ ]:


train_img=src.databunch('.',bs=50)


# In[ ]:


train_img.show_batch()


# ## Transforms
# 
# From the competetion details, we know these images were taken from air. So they resemble satelite imagery. Looking at the images though, they are very low resolution and look very ugly when blown up :D
# 
# Anyways, some features are apparent from looking at the data:
# 1. Some pictures are flipped vertically.
# 2. Some are rotated.
# 3. Some are zoomed in and some aren't.
# 
# Lucky for us, fastai has some default transforms ready. All we need to do is to plug them in. Default transforms include zooms, rotations and lighting.
# I'm just adding vertical flip in to account for aerial imagery.

# In[ ]:


tfms=get_transforms(flip_vert=True)


# In[ ]:


train_img = (src.transform(tfms,size=128)
            .databunch('.',bs=50)
       )


# ## Training Densenet

# In[ ]:


denselearner = cnn_learner(train_img, models.densenet161, metrics=[FBeta(),error_rate, accuracy])


# In[ ]:


denselearner.lr_find()
denselearner.recorder.plot(suggestion=True)


# In[ ]:


lr = 7.5e-03
denselearner.fit_one_cycle(5, slice(lr))


# In[ ]:


denselearner.unfreeze()
denselearner.lr_find()
denselearner.recorder.plot(suggestion=True)


# In[ ]:


denselearner.fit_one_cycle(1, slice(1e-06))


# ## Training Resnet
# Densenet takes quite a lot of time considering this problem is trivial. Bit of an overkill to be honest.
# Let's try Resnet.

# In[ ]:


reslearner = cnn_learner(train_img, models.resnet101, metrics=[FBeta(),error_rate, accuracy])


# In[ ]:


reslearner.lr_find()


# In[ ]:


reslearner.recorder.plot(suggestion=True)


# In[ ]:


lr=9e-3


# In[ ]:


reslearner.fit_one_cycle(5,slice(lr))


# In[ ]:


reslearner.unfreeze()
reslearner.fit_one_cycle(2,slice(1e-6))


# Resnet converges to 100% accuracy very quickly.
# Plotting losses and watching confusion matrix is of no need.

# In[ ]:


interp = ClassificationInterpretation.from_learner(reslearner)
interp.plot_top_losses(9, figsize=(7,6))


# In[ ]:


preds,_ = reslearner.get_preds(ds_type=DatasetType.Test)


# In[ ]:


test_df.has_cactus = preds.numpy()[:, 0]


# In[ ]:


test_df.to_csv('submission.csv', index=False)

