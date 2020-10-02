#!/usr/bin/env python
# coding: utf-8

# ## Fast.ai - ResNet101 with Unfreeze

# ## Make sure you...
# - Have both the data set attached as well as the model (resnet101)
# - You have GPU enabled in settings

# In[ ]:


# Put these at the top of every notebook, to get automatic reloading and inline plotting
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# This file contains all the main external libs we'll use
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
import os
# Let's see what the directories are like
print(os.listdir("../input/"))


# In[ ]:


# After some listdir fun we've determined the proper path
PATH = '../input/dice-d4-d6-d8-d10-d12-d20-images/dice-d4-d6-d8-d10-d12-d20/dice'

# Let's make the resnet101 model available to FastAI
# Credit to Shivam for figuring this out: 
# https://www.kaggle.com/shivamsaboo17/amazon-from-space-using-fastai/notebook AND http://forums.fast.ai/t/how-can-i-load-a-pretrained-model-on-kaggle-using-fastai/13941/7
from os.path import expanduser, join, exists
from os import makedirs
cache_dir = expanduser(join('~', '.torch'))
if not exists(cache_dir):
    makedirs(cache_dir)
models_dir = join(cache_dir, 'models')
if not exists(models_dir):
    makedirs(models_dir)

# copy time!
get_ipython().system('cp ../input/resnet101/resnet101.pth /tmp/.torch/models/resnet101-5d3b4d8f.pth')


# Let's set the initial parameters. If you have less RAM,  we would lower the batch size to **24**, such as on a *GTX 1060 with 6GB GPU memory*.

# In[ ]:


arch=resnet101
workers=8 # This number should match your number of processor cores for best results
sz=240 # image size 240x240
bs=64 # batch size
learnrate = 5e-3 #0.005
dropout = [0.3,0.6] # I found this to be the sweet spot for this data set to reduce overfitting with high accuracy
# I used the following notebook structure to help determine a good rate: 
# https://github.com/ucffool/fastai-custom-learning-notebooks/blob/master/Testing%20Dropout%20Rates%20(small%20images).ipynb


# Let's add some basic transforms (zoom and lighting)

# In[ ]:


# Since this data set already incorporates basic rotations in the set due to the method used, no additional transforms used (the model actually overfits and gets worse)
# tfms = tfms_from_model(arch, sz, max_zoom=1.1)

# TESTING added lighting changes | July 13, 2018 Version 21
tfms = tfms_from_model(arch, sz, aug_tfms = [RandomLighting(b=0.5, c=0.5, tfm_y=TfmType.NO)], max_zoom=1.1)
data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=bs, num_workers=workers)


# In[ ]:


# adjust the path for writing to something writeable (this avoids an error about read-only directories)
import pathlib
data.path = pathlib.Path('.')


# In[ ]:


# Make sure precompute is FALSE or it will not complete when you COMMIT (even if it runs when editing)
learn = ConvLearner.pretrained(arch, data, precompute=False, ps=dropout)


# In[ ]:


# Finding the learning rate
lrf=learn.lr_find()
# Plotting the learning rate
learn.sched.plot()


# Looks like we've got a good learning rate set, so let's continue...

# In[ ]:


learn.fit(learnrate, 1)


# Quick check and we're already doing well! Let's get busy with a few more epochs.

# In[ ]:


get_ipython().run_line_magic('time', 'learn.fit(learnrate, 2, cycle_len=1)')
lr = learnrate # just in case the next code block is skipped


# ### (Optional) Don't Unfreeze
# **For faster results and for comparison to learn.unfreeze() cycles,** *comment  out the below code block (or skip it)*

# In[ ]:


learn.unfreeze()
# BatchNorm is recommended when using anything bigger than resnet34, but on my local test the results were worse so I'll comment it out for now
# learn.bn_freeze(True) 
lr=np.array([learnrate/100,learnrate/10,learnrate])


# In[ ]:


get_ipython().run_line_magic('time', 'learn.fit(lr, 3, cycle_len=1, cycle_mult=2)')
# %time learn.fit(lr, 3, cycle_len=1)


# In[ ]:


learn.save("240_resnet101_all")


# In[ ]:


learn.load("240_resnet101_all")


# In[ ]:


get_ipython().run_line_magic('time', 'log_preds,y = learn.TTA()')
probs = np.mean(np.exp(log_preds),0)
accuracy_np(probs,y)


# ## Analyzing results (99.5% or better)

# In[ ]:


preds = np.argmax(probs, axis=1)
probs = probs[:,1]

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, preds)


# In[ ]:


plot_confusion_matrix(cm, data.classes)


# In[ ]:




