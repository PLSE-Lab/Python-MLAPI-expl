#!/usr/bin/env python
# coding: utf-8

# **Introduction**
# 
# I put this kernel here because [this](https://github.com/ZlodeiBaal/) guy told me about critical issues in **some** other kernels:
# 
# All the data (images from train/val/test folders) are mixed into train/validation distribution randomly. It does not make any sence since there are a lot of images from "train" folder were taken from the same patients. As a result the model is validated on data it was already trained which makes the results completely irrelevant.
# So you cannot use any images from "train" folder for validation if you want to avoid data leakage.
# 
# **The kernel itself**
# 
# Based on the introduction above we'll try to use fast.ai for pneumonia classification.
# Here we start with importing necessary things:

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
from fastai.vision import *
from fastai.metrics import error_rate
import os


# We should find batch size carefully if we don't want run out of GPU memory.

# In[ ]:


bs = 64


# Here is a path to our data

# In[ ]:


path = Path('../input/chest-xray-pneumonia/chest_xray/chest_xray')
path.ls()


# Here we use 'train' folder for training and 'test' folder for validation (in this example we just drop 'val' folder since it has only 16 images). We also resize our images to 299x299 and perform an augmentation with default options.

# In[ ]:


np.random.seed(5)
data = ImageDataBunch.from_folder(path, valid = 'test', size=299, bs=bs, ds_tfms=get_transforms()).normalize(imagenet_stats)


# Let's look into some data

# In[ ]:


data.show_batch(rows=3, figsize=(6,6))


# Let's check classes and amount of data in them.

# In[ ]:


data.classes, data.c, len(data.train_ds), len(data.valid_ds)


# Here is our learner based on ResNet50 architecture pre-trained on ImageNet.

# In[ ]:


learn = cnn_learner(data, models.resnet50, model_dir = '/tmp/model/', metrics=error_rate)


# No we train the model for 8 epochs following [One Cycle Policy](https://arxiv.org/pdf/1803.09820.pdf).

# In[ ]:


learn.fit_one_cycle(8)


# Let's save our weights here.

# In[ ]:


learn.save('step-1-50')


# No we unfreeze the whole model for fine tuning since we trained only last group of layers above.

# In[ ]:


learn.unfreeze()


# Also we find optimal learning rate for fine tuning.

# In[ ]:


learn.lr_find()
learn.recorder.plot()


# Here we train the model for 10 more epochs with different learning rate values for different groups of layers.

# In[ ]:


learn.fit_one_cycle(10, max_lr=slice(7e-6, 3e-4))


# Here we save new weights.

# In[ ]:


learn.save('step-2-50')


# Finally we draw confusion matrix.

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# *Validation Accuracy* = 91.2%
# 
# *Precision* = 386 / (386 + 51) = 88.3%
# 
# *Recall* = 386 / (386 + 4) = 99.0%
