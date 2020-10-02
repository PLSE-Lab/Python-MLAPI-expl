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


# **Introduction:**
# This is based on Fast AI library Lesson 1( Dogs Vs Cats) using resnet 34 pre-trained model with help from A Beginner's Approach to Classification by archaeocharlie & MNIST test with fastai library by Stefan Langenbach
# 

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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[ ]:


#set the path
PATH="../input"
os.listdir(PATH)


# Using Pandas to upload the file

# In[ ]:


train=pd.read_csv(f'{PATH}/train.csv')
test=pd.read_csv(f'{PATH}/test.csv')


# The above training file contains both images and labels.  These have to be split. First column is a label
# 

# In[ ]:


image=train.iloc[:,1:]
lbl=train.iloc[:,0:1]


# To view as image and load into fastai library using image classifier data.  Since pre-trained model resnet has 3 channels, we will have to multiply the channels by 3 the test and train data

# In[ ]:


#Reshape an existing 2D pandas.dataframe into 3D-numpy.ndarray
img=image.as_matrix()
img=img.reshape(-1,28,28)
test_img=test.as_matrix()
test_img=test_img.reshape(-1,28,28)
#Add missing color channels to previously reshaped image
img=np.stack((img,)*3, axis=-1).astype('float32')
test_img=np.stack((test_img,)*3, axis=-1).astype('float32')
  
#plt.imshow(img[2])
#plt.title(lbl.iloc[i,0]);


# In[ ]:


plt.imshow(img[3]);


# In[ ]:


#not required
#convert images into a proper np.ndarray
#img=img.flatten()
#print([i.shape for i in img])


# Split the Training dataset into Train and Valid

# In[ ]:


train_img, val_img, train_lbl, val_lbl=train_test_split(img, lbl, train_size=0.8,random_state=1)


# In[ ]:


grp=[train_img, val_img, train_lbl, val_lbl, test_img]
print([e.shape for e in grp])
print([type(e) for e in grp])


# Since the label is in the form of dataframe, it needs to be converted into array

# In[ ]:


train_lbl=train_lbl.values.flatten()
val_lbl=val_lbl.values.flatten()


# In[ ]:


grp=[train_img, val_img, train_lbl, val_lbl]
print([e.shape for e in grp])
print([type(e) for e in grp])


# In[ ]:


arch=resnet34
sz=28
classes=np.unique(train_lbl)
data=ImageClassifierData.from_arrays(path="/tmp",trn=(train_img/255, train_lbl),
                                     val=(val_img/255, val_lbl),
                                     classes=train_lbl,
                                     test=test_img/255,
                                     tfms=tfms_from_model(arch, sz, max_zoom=1.1))


# In[ ]:


learn=ConvLearner.pretrained(arch, data, precompute=True)


# In[ ]:


learn.lr_find()


# In[ ]:


learn.sched.plot_lr()


# In[ ]:


learn.sched.plot()


# Based on the above plot, we will select 0.01 as the learning rate. Before running the below code, re-run ConvLearner and run the below code

# In[ ]:


learn=ConvLearner.pretrained(arch, data, precompute=True)


# In[ ]:


learn.fit(0.01,9)


# Moving to the second chapter.  We will use cycle rate and data augmentations.  Re-run ConvLearner code again

# In[ ]:


learn=ConvLearner.pretrained(arch, data, precompute=False)


# In[ ]:


#learn.precompute=False
learn.fit(0.01,9, cycle_len=1)


# Cycle_len hasn't improved the accuracy or the loss. Cycle_len enables stochastic gradient descent with restarts (SGDR).  This helps model to jump to the different part in the weight space

# In[ ]:


learn.sched.plot_lr()


# Fine tuning other layers with final layer being trained. 

# In[ ]:


learn.unfreeze()


#  The earlier layers (as we've seen) have more general-purpose features. Therefore we would expect them to need less fine-tuning for new datasets. For this reason we will use different learning rates for different layers: the first few layers will be at 1e-4, the middle layers at 1e-3, and our FC layers we'll leave at 1e-2 as before. We refer to this as differential learning rates,

# In[ ]:


lr=np.array([1e-4,1e-3,1e-2])


# In[ ]:


learn.fit(lr, 4, cycle_len=1, cycle_mult=2)


# Cycle Mult multiplies the length of the cycle after each cycle. e.g. epoch=4, cycle_mult=2 then it multiples the length of the cycle after each cycle (1 epoch + 2 epoch + 4 epoch + 8 epoch=15 epochs)

# In[ ]:


learn.sched.plot_lr()


# In[ ]:


learn.save('4_epochs')


# In[ ]:


#predict the test set
get_ipython().run_line_magic('time', 'log_preds_test, y_test=learn.TTA(is_test=True)')
probs_test=np.mean(np.exp(log_preds_test),0)
probs_test.shape


# To create a submission file

# In[ ]:


#Create a dataframe from all probabilities
df=pd.DataFrame(probs_test)


# In[ ]:



df.head()


# In[ ]:


#consider the maxm probability

df=df.assign(Label=df.values.argmax(axis=1))
df=df.assign(ImageId=df.index.values+1)


# In[ ]:


df1=df[['ImageId', 'Label']]


# In[ ]:


df1.head()


# In[ ]:


df1.shape


# In[ ]:


df1.to_csv("submission.csv", index=False)

