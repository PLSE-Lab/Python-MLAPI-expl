#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Put these at the top of every notebook, to get automatic reloading and inline plotting
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # showing and rendering figures
# io related
from skimage.io import imread
import os
from glob import glob

from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *


# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input/dr_data/DR_data/"]).decode("utf8"))


# In[ ]:


PATH ="../input/dr_data/DR_data/"
TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"
arch=resnet34
sz=224
tfms=tfms_from_model(arch, sz)


# In[ ]:


torch.cuda.is_available()


# In[ ]:


torch.backends.cudnn.enabled


# In[ ]:


fnames = np.array([f'train/{f}' for f in sorted(os.listdir(f'{PATH}train'))])


# In[ ]:


img = plt.imread(f'{PATH}{fnames[0]}')
plt.imshow(img);


# Here is how the raw data looks like

# In[ ]:


img.shape


# In[ ]:


img[2000:2004,1500:1504]


# ## Our first model: quick start

# In[ ]:


# Uncomment the below if you need to reset your precomputed activations
# shutil.rmtree(f'{PATH}tmp', ignore_errors=True)


# In[ ]:


data= ImageClassifierData.from_csv(path=PATH,
                                      folder='train', 
                                      csv_fname='../input/trainLabels_3.csv'
                                      , tfms=tfms, test_name='test', 
                                       suffix='.jpeg')
learn = ConvLearner.pretrained(arch, data, precompute=True, tmp_name=TMP_PATH, models_name=MODEL_PATH)
learn.fit(0.01, 2)


# In[ ]:


data


# ## Analyzing results: looking at pictures

# In[ ]:


# This is the label for a val data
data.val_y


# In[ ]:


# from here we know that 'cats' is label 0 and 'dogs' is label 1.
data.classes


# In[ ]:


# this gives prediction for validation set. Predictions are in log scale
log_preds = learn.predict()
log_preds.shape


# In[ ]:


log_preds[:10]


# In[ ]:


preds = np.argmax(log_preds, axis=1)  # from log probabilities to 0 or 1
probs = np.exp(log_preds[:,0])        # pr(no DR)


# In[ ]:


def rand_by_mask(mask): return np.random.choice(np.where(mask)[0], 4, replace=False)
def rand_by_correct(is_correct): return rand_by_mask((preds == data.val_y)==is_correct)


# In[ ]:


def plots(ims, figsize=(12,6), rows=1, titles=None):
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i])


# In[ ]:


def load_img_id(ds, idx): return np.array(PIL.Image.open(PATH+ds.fnames[idx]))

def plot_val_with_title(idxs, title):
    imgs = [load_img_id(data.val_ds,x) for x in idxs]
    title_probs = [probs[x] for x in idxs]
    print(title)
    return plots(imgs, rows=1, titles=title_probs, figsize=(16,8))


# In[ ]:


# 1. A few correct labels at random
plot_val_with_title(rand_by_correct(True), "Correctly classified")


# In[ ]:


# 2. A few incorrect labels at random
plot_val_with_title(rand_by_correct(False), "Incorrectly classified")


# In[ ]:


def most_by_mask(mask, mult):
    idxs = np.where(mask)[0]
    return idxs[np.argsort(mult * probs[idxs])[:4]]

def most_by_correct(y, is_correct): 
    mult = -1 if (y==1)==is_correct else 1
    return most_by_mask(((preds == data.val_y)==is_correct) & (data.val_y == y), mult)


# In[ ]:


plot_val_with_title(most_by_correct(0, True), "Most correct No Retinopathy")


# In[ ]:


plot_val_with_title(most_by_correct(4, True), "Most correct Retinopathy")


# In[ ]:


plot_val_with_title(most_by_correct(0, False), "Most incorrect No Retinopathy")


# In[ ]:


plot_val_with_title(most_by_correct(1, False), "Most incorrect Retinopathy")


# In[ ]:


most_uncertain = np.argsort(np.abs(probs -0.5))[:4]
plot_val_with_title(most_uncertain, "Most uncertain predictions")

