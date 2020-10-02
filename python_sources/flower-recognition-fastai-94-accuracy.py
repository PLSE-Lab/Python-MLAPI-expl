#!/usr/bin/env python
# coding: utf-8

# #### Flower Classification
# 
# List of classes of Flowers given in the dataset-
# 1. Daisy
# 2. Rose
# 3. Dandelion
# 4. Tulip
# 5. Sunflower
# 
# Idea here is to fine-tune a pretrained model (**Resnet34**) using the FastAI Library to get the best possible (close to SOTA) result.

# ### Necessary Library Imports
# 
# A directory containing pretrained **Resnet34** model was also required and was available on Kaggle as a public dataset.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pathlib

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import gc
print(os.listdir("../input"))

from sklearn.metrics import confusion_matrix
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

# Any results you write to the current directory are saved as output.


# The following are the two helper modules required to put Resnet34 weights in the apt directory for PyTorch to use directly. These were taken from Anshul Rai's [kernel here](https://www.kaggle.com/anshulrai/using-fastai-in-kaggle-kernel).

# In[ ]:


cache_dir = os.path.expanduser(os.path.join('~', '.torch'))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
models_dir = os.path.join(cache_dir, 'models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)


# In[ ]:


get_ipython().system('cp ../input/resnet34/resnet34.pth /tmp/.torch/models/resnet34-333f7ec4.pth')


# In[ ]:


PATH = '../input/flowers-train-valid-split/flowers_split/flowers_split/'
sz=224


# In[ ]:


os.listdir(f'{PATH}valid')


# ### Check status of GPU Availability

# In[ ]:


torch.cuda.is_available()


# In[ ]:


sample = os.listdir(f'{PATH}valid/daisy')[:5]
#sample


# In[ ]:


img = plt.imread(f'{PATH}valid/daisy/{sample[0]}')
plt.imshow(img);
del sample


# ### Image Dimensions
# 
# Here, we have got ourselves a standard **3 Channel** image so our pretrained models should work fine with added tricks of Data Augmentation.

# In[ ]:


#img.shape
del img


# ## Baseline Model (88% Accuracy)
# 
# - Resnet34 Architecture
# - Precomputed Activations
# - No Data Augmentation

# In[ ]:


arch = resnet34
data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch,224))
data.path = pathlib.Path('.')  ## IMPORTANT for PyTORCH to create tmp directory which won't be otherwise allowed on Kaggle Kernel directory structure
learn = ConvLearner.pretrained(arch, data, precompute=False) #Precompute=True causes the Commit & run operation to fail
learn.fit(0.01, 2)


# In[ ]:


gc.collect()
data.classes


# In[ ]:


log_preds = learn.predict()
log_preds.shape
preds = np.argmax(log_preds, axis=1)
probs = np.exp(log_preds[:,1]) 


# In[ ]:


def rand_by_mask(mask): return np.random.choice(np.where(mask)[0], 4, replace=False)
def rand_by_correct(is_correct): return rand_by_mask((preds == data.val_y)==is_correct)

def plot_val_with_title(idxs, title):
    imgs = np.stack([data.val_ds[x][0] for x in idxs])
    title_probs = [probs[x] for x in idxs]
    print(title)
    return plots(data.val_ds.denorm(imgs), rows=1, titles=title_probs)

def plots(ims, figsize=(12,6), rows=1, titles=None):
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i])

def load_img_id(ds, idx): return np.array(PIL.Image.open(PATH+ds.fnames[idx]))

def plot_val_with_title(idxs, title):
    imgs = [load_img_id(data.val_ds,x) for x in idxs]
    title_probs = [probs[x] for x in idxs]
    print(title)
    return plots(imgs, rows=1, titles=title_probs, figsize=(16,8))


# In[ ]:


plot_val_with_title(rand_by_correct(True), "Correctly classified")


# In[ ]:


def most_by_mask(mask, mult):
    idxs = np.where(mask)[0]
    return idxs[np.argsort(mult * probs[idxs])[:4]]

def most_by_correct(y, is_correct): 
    mult = -1 if (y==1)==is_correct else 1
    return most_by_mask(((preds == data.val_y)==is_correct) & (data.val_y == y), mult)


# In[ ]:


plot_val_with_title(most_by_correct(2, True), "Most correct Roses")


# ## Finding a Learning Rate
# 
# Using the `lr_find` (learning rate finder) from the FastAI library to get an optimum learning rate.

# In[ ]:


lrf=learn.lr_find()


# ### Learning Rate ~ 0.01

# In[ ]:


learn.sched.plot()


# In[ ]:


del data
del learn


# In[ ]:


tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_paths(PATH, tfms=tfms)
data.path = pathlib.Path(".")
learn = ConvLearner.pretrained(arch, data, precompute=False)
learn.fit(1e-2, 1)
gc.collect()


# ## Best Model (93% Accuracy)
# 
# - **No precomputed activations.**
# - **Unfreezing all layers.**
# - **Use of SGDR with varying Learning Rates for each set of layers.**

# In[ ]:


learn.precompute = False
learn.unfreeze()
lr=np.array([1e-4,1e-3,1e-2])
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)
#learn.sched.plot_lr()


# In[ ]:


log_preds,y = learn.TTA()
probs = np.mean(np.exp(log_preds),0)


# # Create a confusion matrix to visualize class-wise results!

# In[ ]:


accuracy_np(probs, y)


# In[ ]:


#plt.figure(figsize=(15,15))
preds = np.argmax(probs, axis=1)
probs = probs[:,1]
cm = confusion_matrix(y, preds)


# ## Confusion Matrix (Dev Set)

# In[ ]:


plot_confusion_matrix(cm, data.classes, figsize=(10,10))


# In[ ]:




