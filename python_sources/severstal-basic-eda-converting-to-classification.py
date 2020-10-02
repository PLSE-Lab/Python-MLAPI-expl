#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Welcome to the Severstal: Steel Defect Detection competition. This competition is a two-fold competition: classify the type of steel defect, and also segment the parts of the image that contain the defect.
# 
# In this kernel, I will do a quick EDA and then I will convert to a classification problem for future kernels.
# 

# ## EDA

# In[ ]:


import numpy as np
import pandas as pd
import os
from fastai.vision import *


# In[ ]:


print(os.listdir('../input/severstal-steel-defect-detection'))


# Let's look at the training set csv file:

# In[ ]:


comp_dir = Path('../input/severstal-steel-defect-detection')


# In[ ]:


train_df = pd.read_csv(comp_dir/'train.csv')
train_df.head(20)


# Looking at this we can see that the format is to fill in the EncodedPixels columens only at the rows  for the identified class for the selected image. For example, 0002cc93b.jpg has a label of 1, 00031f466.jpg has no defects, and 0007a71bf.jpg has a label of 3.

# ### Conversion to classification
# 
# I believe that one possible strategy for this competition would be to perhaps first create a classifier and have separate segmentation models for each of the defect types. As a first step to do this, I reformat the data, converting it into a multi-label classification problem.

# Lazy conversion to classification labels:

# In[ ]:


labels = []
for i in range(len(train_df)):
    if type(train_df.EncodedPixels[i]) == str:
        labels.append(1)
    else:
        labels.append(0)
labels = np.array(labels)
labels = labels.reshape((int(len(train_df)/4),4))


# In[ ]:


print(labels.shape)


# In[ ]:


label_0 = np.array(len(labels) - np.sum(np.sum(labels,axis=0)))
bar_plot = np.append(label_0[None].T,np.sum(labels,axis=0))
plt.bar(np.array(['none','0','1','2','3']),bar_plot)


# 1. Most common is no defect, and defect of type 2. In fact we see that 10% of the labels have no label and 10% have are label of type 2:
# 
# 

# In[ ]:


label_0/len(train_df)


# In[ ]:


bar_plot[3]/len(train_df)


# Let's now convert this to something fastai can view.

# In[ ]:


images_df = pd.DataFrame(train_df.iloc[::4,:].ImageId_ClassId.str[:-2].reset_index(drop=True))
labels_df = pd.DataFrame(labels.astype(int))


# In[ ]:


proc_train_df= pd.concat((images_df,labels_df),1)
proc_train_df


# Let's load into fastai:

# In[ ]:


data = (ImageList.from_df(proc_train_df,path=comp_dir,folder='train_images')
        .split_by_rand_pct(0.2)
        .label_from_df(cols=[1,2,3,4])
        .transform(get_transforms())
        .databunch(bs=16)
       )


# In[ ]:


#data.show_batch()


# In[ ]:


data.train_ds[0][0].shape


# We can see that the images have a fixed size of 256x1600.

# ## Training a classifier
# 
# Let's create a classifier using fastai.

# In[ ]:


# Making pretrained weights work without needing to find the default filename
if not os.path.exists('/tmp/.cache/torch/checkpoints/'):
        os.makedirs('/tmp/.cache/torch/checkpoints/')
get_ipython().system("cp '../input/resnet50/resnet50.pth' '/tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth'")


# In[ ]:


from fastai.metrics import *
learn = cnn_learner(data,models.resnet50,metrics=accuracy_thresh)
learn.model_dir = Path('../models')


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(1,2e-2)


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(5,slice(1e-6,1e-3))


# In[ ]:


print(learn.validate())

