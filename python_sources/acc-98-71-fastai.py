#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This notebook applies fast.ai course v3 [lesson1](https://course.fast.ai/videos/?lesson=1) to the [Planes in Satellite Imagery dataset](https://www.kaggle.com/rhammell/planesnet).
# We will use a ResNet50 architecture with some transfer learning first, and then update the weight of the whole architecture.
# The model will have 2 outputs, planes (1) or no-planes (0).
# 
# This achieved an accuracy of 98.71.
# 
# Also, if you look at the final plot of the top losses, you will find out that even for a human it is very difficult (impossible) to say if there is a plane or not.
# Knowing that the data has been manually labelled let me think that those might even have been mislabeled...

# ## Setup

# Ensure that:
# * any edits to libraries are reloaded automatically
# * any charts or images displayed are shown in this notebook

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# Import necessary packages

# In[ ]:


from fastai.vision import *
from fastai.metrics import error_rate


# In[ ]:


import fastai
print('fastai version :', fastai.__version__)


# In[ ]:


# Batch Size (adequate size for GPU of 11GB or more)
bs = 64


# ## Data Bunch creation

# In[ ]:


path = Path('../input/planesnet/planesnet/planesnet')
# path.ls()


# In[ ]:


fnames = get_image_files(path)
fnames[:5]


# In[ ]:


# regex to extract category
pat = r'^\D*(\d+)'


# For reproducibility

# In[ ]:


np.random.seed(23)


# In[ ]:


# Setup the transformations to apply to the training data
tfms = get_transforms(flip_vert=True)

# Add the images to the Image Data Bunch
data = ImageDataBunch.from_name_re(path, fnames, pat, ds_tfms=tfms, size=21, bs=bs).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(7,6))


# In[ ]:


print(data.classes)
len(data.classes),data.c


# ## Model Creation - Transfer Learning with ResNet50

# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=error_rate, model_dir="/output/kaggle/working/model")


# In[ ]:


learn.model


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.save('stage-1', return_path=True)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses, idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


interp.most_confused(min_val=2)


# ## Model Update - Retraining of all the layer of the ResNet50

# In[ ]:


learn.unfreeze()


# In[ ]:


lr_find(learn)
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(6, slice(1e-5,1e-3))


# In[ ]:


learn.save('stage-2', return_path=True)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses, idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


interp.most_confused(min_val=2)

