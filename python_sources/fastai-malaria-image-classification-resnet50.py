#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fastai import *
from fastai.vision import *
from fastai.vision.gan import *
from fastai.callbacks.hooks import *

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input/cell_images/cell_images"))

# Any results you write to the current directory are saved as output.


# ## Comprising the Dataset
# 
# For this project, the dataset has been split into two separate folders within cell_images:
# 
#     * Parasitized
#     * Uninfected
# 
# where parasitized is those cell images to which have been infected with malaria.

# In[ ]:


img_dir='../input/cell_images/cell_images/'
path = Path(img_dir)


# With the path defined to the dataset, we can start comprising the [ImageDataBunch](http://https://docs.fast.ai/vision.data.html#ImageDataBunch) which will form the data we will propagate over throughout the course of this notebook.

# In[ ]:


data = ImageDataBunch.from_folder(path, 
                                  train=".",
                                  valid_pct=0.2, 
                                  ds_tfms=get_transforms(flip_vert=True, max_warp=0),
                                  size=224,bs=64, 
                                  num_workers=0).normalize(imagenet_stats)


# Let's look at some of the images to check that the data has loaded correctly.

# In[ ]:


data.show_batch(rows=5)


# In[ ]:


print(f'Classes: \n {data.classes}')


# ## Model Development
# 
# For image classification, we shall utilise a commonly used model for image classification, Resnet50 to provide us with a ready made means to start work.
# To compare epochs, we will utilise two metrics to determine validity: 
# 
#     * Accuracy
#     * Error rate
#     
# Finally, the model will save its progress in '/temp/model'. This is due to a limitation where the folder directory used for the project is set to read only.

# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=[accuracy, error_rate], model_dir="/temp/model/")


# With the model defined, it serves us to learn the most effective hyperparameters we shall utilise for the model. For this, we shall use [lr_find](http://https://docs.fast.ai/basic_train.html#lr_find) to find a good learning rate for the model.

# In[ ]:


learn.lr_find()
learn.recorder.plot()


# Looks like a learning rate of 1e-01 is a good starting point for the model. We can now start learning!
# We will utilise the above learning rate through 5 epochs, or cycles, and compare the results.

# In[ ]:


learn_rate = 1e-01


# In[ ]:


learn.fit_one_cycle(5, slice(learn_rate))


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix(figsize=(8,8), dpi=60)


# Superb performance thus far, but there are a few pesky outliers that we could look to recapture. The model is at least failing at points which make sense at this stage, so we can continue.
# We can save the model for future work.

# In[ ]:


learn.save('stage-1-rn50')


# Before we start trying to optimise the model further, we need to unfreeze it to continue work. Thankfully, FastAI allows you to do that with a simple function call, unfreeze()

# In[ ]:


learn.unfreeze()


# With the model good to go, we can attempt to find a better learning rate now the model has a better representation of the data.

# In[ ]:


learn.lr_find()
learn.recorder.plot()


# Looks like a learning rate of 1e-05 is a better learning rate for the model.
# We will utilise the above learning rate through 5 epochs, or cycles, and compare the results.

# In[ ]:


learn.fit_one_cycle(5, slice(1e-5, learn_rate/5))


# In[ ]:


learn.save('stage-2-rn50')


# In[ ]:


learn.unfreeze()
learn.recorder.plot_losses()


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix(figsize=(8,8), dpi=60)


# Overall, a well performing image classifier has been devised for this particular task. However, it is prudent to see exactly where the image classifier is looking when it is attempting to differentiate between the classes.

# ## Review Model
# 
# We can utilise a heatmap, provided by [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](http://https://arxiv.org/abs/1610.02391) to see exactly where the model is looking when it is attempting to differentiate between the classes. 
# 

# In[ ]:


from fastai.callbacks.hooks import *

idx=0
x,y = data.valid_ds[idx]
eval_model = learn.model.eval();
xb,_ = data.one_item(x)
xb_im = Image(data.denorm(xb)[0])
xb = xb.cuda()


# In[ ]:


def hooked_backward(cat=y):
    with hook_output(eval_model[0]) as hook_a: 
        with hook_output(eval_model[0], grad=True) as hook_g:
            preds = eval_model(xb)
            preds[0,int(cat)].backward()
    return hook_a, hook_g


# In[ ]:


def show_heatmap(hm):
    _,ax = plt.subplots()
    xb_im.show(ax)
    ax.imshow(hm, alpha=0.6, extent=(0,352,352,0),
              interpolation='bilinear', cmap='magma');


# In[ ]:


hook_a, hook_g = hooked_backward()


# In[ ]:


acts  = hook_a.stored[0].cpu()
acts.shape


# In[ ]:


grad = hook_g.stored[0][0].cpu()
grad_chan = grad.mean(1).mean(1)
grad.shape,grad_chan.shape


# In[ ]:


mult = (acts*grad_chan[...,None,None]).mean(0)
show_heatmap(mult)


# Got to admit, it's quite cool to see where it's seeing!
