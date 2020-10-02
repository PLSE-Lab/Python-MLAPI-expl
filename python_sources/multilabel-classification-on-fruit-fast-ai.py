#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.conv_learner import *


# In[ ]:


PATH = '../input/fruits-360_dataset/'


# This is to kept as a reference of the folder structure

# In[ ]:


read_path = '../input/fruits-360_dataset/*'


# In[ ]:


ls {read_path}


# In[ ]:


ls {PATH}


# In[ ]:


from fastai.plots import * 


# In[ ]:


def get_1st(path, pattern): return glob(f'{path}/*{pattern}.*')[2]


# In[ ]:


list_paths = [f"{PATH}fruits-360/Training/Apple Braeburn/0_100.jpg", f"{PATH}fruits-360/Training/Apple Golden 1/116_100.jpg"]
titles = ["Apple Braeburn", "Apple Golden 1" ]
plots_from_files(list_paths,titles=titles,maintitle="Multi-label clasification")


# Import the planet.py 
# 

# In[ ]:


# the planet.py file

from fastai.imports import *
from fastai.transforms import *
from fastai.dataset import *
from sklearn.metrics import fbeta_score
import warnings

def f2(preds, targs, start=0.17, end=0.24, step=0.01):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return max([fbeta_score(targs, (preds>th), 2, average='samples')
                   for th in np.arange(start,end,step)])


# In[ ]:


metrics=[f2]
f_model = resnet34


# In[ ]:


# import the training dataset
import glob 
import cv2

training_fruit_img = []
training_label = []
for dir_path in glob.glob("../input/*/fruits-360/Training/*"):
    img_label = dir_path.split("/")[-1]
    for img_path in glob.glob(os.path.join(dir_path, "*.jpg")):
        img = cv2.imread(img_path)
        training_fruit_img.append(img)
        training_label.append(img_label)
training_fruit_img = np.array(training_fruit_img)
training_label = np.array(training_label)
len(np.unique(training_label))


# This step is to see the different classes in our dataset

# In[ ]:


label_to_id_dict = {v:i for i,v in enumerate(np.unique(training_label))}
id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}


# In[ ]:


id_to_label_dict


# In[ ]:


label_ids = np.array([label_to_id_dict[x] for x in training_label])


# In[ ]:


label_ids.shape,training_label.shape


# In[ ]:


training_path = f'{PATH}fruits-360/'
def get_data(sz):
    tfms = tfms_from_model(f_model, sz, aug_tfms=transforms_top_down, max_zoom=1.05)
    return ImageClassifierData.from_paths(path=training_path,trn_name="Training", val_name="Test", tfms=tfms)


# In[ ]:


data = get_data(256)


# In[ ]:


y


# In[ ]:


plt.imshow(data.val_ds.denorm(to_np(x))[54]*1.4)


# # Find Learning Rate

# In[ ]:


sz=64
data = data.resize(int(sz*1.3), '/tmp')


# In[ ]:


arch = resnet34


# In[ ]:


learn = ConvLearner.pretrained(arch,data, precompute=True)


# In[ ]:


lrf=learn.lr_find()
learn.sched.plot()


# In[ ]:



learn.fit(lrs=0.01,n_cycle=3)


# In[ ]:


learn.sched.plot_loss()


# In[ ]:


log_preds = learn.predict()
preds = np.argmax(log_preds, axis=1)


# In[ ]:


preds


# In[ ]:


probs = np.exp(log_preds[:,1])


# # Lets view our results

# In[ ]:


def rand_by_mask(mask): return np.random.choice(np.where(mask)[0], 4, replace=False)
def rand_by_correct(is_correct): return rand_by_mask((preds == data.val_y)==is_correct)


# In[ ]:



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
        
def load_img_id(ds, idx): return np.array(PIL.Image.open(training_path+ds.fnames[idx]))

def plot_val_with_title(idxs, title):
    imgs = [load_img_id(data.val_ds,x) for x in idxs]
    title_probs = [probs[x] for x in idxs]
    print(title)
    return plots(imgs, rows=1, titles=title_probs, figsize=(16,8))


# In[ ]:


plot_val_with_title(rand_by_correct(True), "Correct Predictions")


# In[ ]:


plot_val_with_title(rand_by_correct(False), "Wrong Predictions")


# In[ ]:



def  most_by_mask(mask, mult):
    idxs = np.where(mask)[0]
    return idxs[np.argsort(mult * probs[idxs])[:4]]

def most_by_correct(y, is_correct): 
    mult = -1 if (y==1)==is_correct else 1
    return most_by_mask((preds == data.val_y)==is_correct & (data.val_y == y), mult)


# In[ ]:


i = 0
plot_val_with_title(most_by_correct(i,True), "Most Correct" + data.classes[i])


# In[ ]:


i = 10
plot_val_with_title(most_by_correct(i,True), "Most Correct" + data.classes[i])


# In[ ]:


i = 15
plot_val_with_title(most_by_correct(i,True), "Most Correct" + data.classes[i])


# In[ ]:


i = 68
plot_val_with_title(most_by_correct(i, True), "Most Correct " + data.classes[i])

