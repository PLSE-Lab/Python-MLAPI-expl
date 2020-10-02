#!/usr/bin/env python
# coding: utf-8

# In this notebook, we will explore the data itself with the help of the `fastai` library. Apart from the state-of-the-art models, it provides a number of handy utilities that are nothing less than golds for a practitioner. 

# In[ ]:


# Depedencies
import numpy as np 
import pandas as pd 
from collections import Counter
import os
print(os.listdir("../input"))


# In[ ]:


train_df = pd.read_csv('../input/train_relationships.csv')
train_df.head()


# In[ ]:


train_df.shape


# As per the data description provided:
# > train.csv - training labels. Remember, not every individual in a family shares a kinship relationship. For example, a mother and father are kin to their children, but not to each other.
# 
# [Shrey Dabhi](https://www.kaggle.com/sdabhi23) further clarified it in [this](https://www.kaggle.com/c/recognizing-faces-in-the-wild/discussion/92238#latest-531309) discussion thread:
# > The persons `p1` and `p2` in a given row are blood relatives of each other. I guess this is provided because not everyone in a family is a blood relative of one another, and a very good example is given in the description itself!

# In[ ]:


# Number of families in the train folder
train = get_ipython().getoutput('ls ../input/train/')
len(train)


# In[ ]:


# Number of unlabelled images
test = get_ipython().getoutput('ls ../input/test/')
len(test)


# In[ ]:


# fastai and torch imports
import torch
from fastai.vision import *
from fastai.metrics import *

np.random.seed(7)
torch.cuda.manual_seed_all(7)


# In[ ]:


# Looking at the naming conventions
train_folder = Path('../input/train')
train_folder.ls()


# In[ ]:


# Looking at specific folder
specific_folder = train_folder/'F0768'
specific_folder.ls()


# In[ ]:


# Looking at individual images belonging to a particular folder
more_specific = train_folder/'F0768/MID4'
more_specific.ls()


# In[ ]:


sample1 = open_image('../input/train/F0768/MID4/P08113_face1.jpg')
show_image(sample1)


# In[ ]:


sample2 = open_image('../input/train/F0768/MID4/P08114_face1.jpg')
show_image(sample2)


# In[ ]:


sample3 = open_image('../input/train/F0768/MID4/P12113_face2.jpg')
show_image(sample3)


# This seems a bit odd :/
# 
# From the data description: 
# > the training set is divided in Families (F0123), then individuals (MIDx). Images in the same MIDx folder belong to the same person. Images in the same F0123 folder belong to the same family.
# 
# The above three sample images are of the same MIDx folder from `F0768`. But the last sample is different from the other two samples :/

# The idea now is to construct a dataset which will represent image to image mappings. We will take each of the images from the folders listed in the `p1` column of the training labels that are provided and will annotate them using the images which will extracted from the `p2` column. Our dataset should be similar to the following:
# 
# ![](https://i.ibb.co/JcbWrTB/Screenshot-from-2019-05-18-19-07-35.png)
# 
# Where the left image is serving as the feature vector and the right image is its label. 

# In order to accomplish this, we first discard the entries that are there in the training csv file but in reality they do exist. Ideally we can take either of columns (p1 and p2) for this and do this. 

# In[ ]:


a = []
for i in train_df.p1:
    try:
        i2=i
        i = Path('../input/train/'+i)
        a.append(i.ls())
    except:
        index_to_drop = train_df.p1[train_df.p1==i2].index.tolist()
        # print(index_to_drop)
        train_df.drop(train_df.index[index_to_drop], inplace=True)

len(train_df), len(a)


# Makes sense, since the list `a` contains lists of image paths. Sorry about the naming conventions, though. 

# In[ ]:


first_person = pd.DataFrame(train_df.p1)
second_person = pd.DataFrame(train_df.p2)
len(first_person)==len(second_person)


# In[ ]:


print(first_person.head(3))
print('\n')
print(second_person.head(3))


# We now construct separate DataFrames for the features and labels since `fastai`'s `ImageDataBunch`es can be created from DataFrames. 

# In[ ]:


# Features DataFrame
a = []
for i in first_person.p1:
    # Suspicious code block since there should not be any FileNotFoundError now
    try:
        i = Path('../input/train/'+i)
        a.append(i.ls())
    except:
        pass

b = []
for i in a:
    for ii in i:
        b.append(ii)
        
features = pd.DataFrame()
features['Path'] = b
features.head()


# In[ ]:


# Labels DataFrame
a = []
for i in second_person.p2:
    try:
        i = Path('../input/train/'+i)
        a.append(i.ls())
    except:
        pass

b = []
for i in a:
    for ii in i:
        b.append(ii)        
labels = pd.DataFrame()
labels['Labels'] = b
labels.head()


# In[ ]:


len(features), len(labels)


# This mismatch is bound to happen.

# In[ ]:


features_new = features[:16307]
features_new['Labels'] = labels['Labels']
features_new.head()


# In[ ]:


features_databunch = ImageImageList.from_df(features_new, path='.')
len(features_databunch)


# In[ ]:


img = open_image(features_databunch.items[0])
img.shape


# In[ ]:


open_image(features_databunch.items[0])


# Referring to the following code block as shown by Jeremy during Lesson 7 (v3, part I):
# ```python
# data = (src.label_from_func(lambda x: path_hr/x.name)
#            .transform(get_transforms(max_zoom=2.), size=size, tfm_y=True)
#            .databunch(bs=bs).normalize(imagenet_stats, do_y=True))
# ```

# Instead of `label_from_func` I am using `label_from_df` since I already have that in a nice format. 

# In[ ]:


databunch = features_databunch.split_by_rand_pct(0.1, seed=7)        .label_from_df(cols='Labels')        .transform(get_transforms(), size=224, tfm_y=True)        .databunch(bs=64).normalize(imagenet_stats, do_y=True)


# In[ ]:


databunch.show_batch(rows=4, figsize=(8,8))


# In[ ]:


learner = unet_learner(databunch, models.resnet34, wd=1e-3, blur=True, norm_type=NormType.Weight,
                            y_range=(-3.,3.), loss_func=MSELossFlat()).to_fp16()
learner.lr_find()
learner.recorder.plot()


# In[ ]:


learner.fit_one_cycle(2, pct_start=0.8, max_lr=slice(1e-05, 1e-03))


# In[ ]:


learner.unfreeze()
learner.fit_one_cycle(2, slice(1e-5,1e-3))


# In[ ]:


learner.validate()

