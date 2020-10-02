#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# # Getting data for Fastai

# In[2]:


# importing our dependencies / Packages
from fastai import *
from fastai.vision import *
import pandas as pd
import numpy as np


# In[3]:


# viewing our data
data_folder = Path("../input")
data_folder.ls()


# ## in this competion the train images have been kept in train folder and its labels kept in /train.csv file. 
# 
# ### further in /sample_submission.csv contains sample submissions

# In[4]:


# Getting the data
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/sample_submission.csv")


# ## Let's create imagelist in this example.
# #### First create test_img which contains list of images in ImageList
# #### then train_img containing training images in ImageList

# In[5]:


test_img = ImageList.from_df(test_df, # test data frame
                             path=data_folder/'test', 
                             folder='test' 
                            )

train_img = (
    ImageList.from_df(train_df, #train data frame
                      path=data_folder/'train', 
                      folder='train')
        .split_by_rand_pct(0.2) # making 20% of validation dataset
        .label_from_df() # it labels according to dataframe passed to it
        .add_test(test_img) # adding test data
        .transform(get_transforms(flip_vert=True), size=128) # adding transforms
        .databunch(path='.', bs=64) # create databunch // path is used internally to store temporary files // bs = batch size
        .normalize(imagenet_stats) #  normalise according to pretained model
       )


# In[6]:


learn = cnn_learner(train_img, #training data
                    models.resnet34,#model
                    metrics=[error_rate, accuracy] #error rate
                   )


# In[ ]:


learn.fit_one_cycle(4) #learning for 4 epochs


# In[ ]:


learn.save('model-1') # saving the model


# # Interpretaion
# Lets understand our model and data and why it has given less accurcay

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


# for plotting top losses
losses,idxs = interp.top_losses() 
interp.plot_top_losses(9,figsize=(15,11))


# In[ ]:


# to view confusion matrix of model
interp.plot_confusion_matrix()


# # UNFREEZING
# Basically in a pretrained model you are freezing the earlier layers by making the weights unchangeable. This is so it can retain the already learned basic representations. The later layers are not frozen so that they can learn the representations specific to your task

# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(1)


# ## Now we have found out in unfreeze model we have found out we got greater accuracy than freeze one, otherwise we have used the freeze one model for analysis

# ### Finding Learning Rate

# We use the lr_find method to find the optimum learning rate. Learning Rate is an important hyper-parameter to look for.

# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# #### From above graph we see different learning rate in which our model perform. choose accordingly.
# 
# As example in cats vs dogs we need to lower learning rate for earlier layers and higher one for the last layers.
# but in this test i think to keep one

# In[ ]:


learn.fit_one_cycle(4, max_lr=slice(1e-05))


# In[ ]:


learn.validate()


# # Prediction

# In[ ]:


preds,_ = learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


test_df.has_cactus = preds.numpy()[:, 0]


# In[ ]:


test_df.to_csv('submission.csv', index=False)

