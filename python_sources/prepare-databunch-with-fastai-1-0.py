#!/usr/bin/env python
# coding: utf-8

# In this notebook, I will prepare whale image with Fast AI datablock API. Addressing the train - validation split problem that many might experience using FastAI 1.0 functions. 
# 
# Also, I hope this kernel can help starters like me to take advantage of the Fast AI and lay down baseline model. 
# 
# I am still learning Fast AI and pytorch, please feel free to leave me a comment as I will be learned from my mistakes :) 
# 
# For people that are interested, here is the link to the Fast AI documentation 
# [fastai doc](https://docs.fast.ai/data_block.html)

# In[ ]:


import numpy as np
import pandas as pd
import os
from fastai.vision import *
from fastai.basic_data import *


# # Take a look of the label data

# In[ ]:


df_label = pd.read_csv('../input/train.csv')
df_label.describe()


# I wont be talking to much about data exploration, as there are already many great kernels talk in depth about the data. 
# 
# There are just couple things people might fall in trap later on when creating databunch object
# 
# 1. There are 5005 unique labels, that means 5005 classes if we load them all into Fast AI. 
# 2. We will have rare labels that we only have 1 image, as well 2 images...3 images as so on. 

# # **Using FastAI data block API**
# The defualt factory method won't work in this case, as it will be hard for you to manually put the data into imageNet style, also, it seems to me very time consuming if you try to create 5005 folders and manually separate each class.
# 
# Therefore we can use data block API, following the documation, it will just be 4 steps
# 
# 1. Provide inputs
# 2. Split data
# 3. Label data
# 4. Create databunch to pass to pytorch

# Here is what we can do
# 1. It is vision data, we can create a ImageItemList object
# 2. We can do random split, since it is a baseline model. 
# 3. We have labels in the train.csv files, and label happens to be col 1, so we can call the default function to handle for as 

# In[ ]:


src = (ImageItemList.from_csv('../input/','train.csv',folder='train')
        .random_split_by_pct()
        .label_from_df())


# ***Exception: Your validation data contains a label that isn't present in the training set, please fix your data***
# 
# Major block people might have using Fast AI libarary is during train valid split (it happens to me at least)
# 
# If you call random_split_by_pct(), default will split the data randomly with 80-20. There will be cases that the class with only 1 image splitted into validation set, now your validation set has a class that train set never knows. Therefore you have the FastAI complaining, because the model can't predict that class since the model never sees it during training. 
# 
# Whale image data set is different as other datasets talked in the class. If you treat it as classfication problem, we are  actually trying to classify an item from 5005 different classes. A good validation set is very important, but we are not addressing it here since we just want to get a baseline model. 

# # Novice way to handle 
# 
# We already know what problem we have, validation set has classes that train set doesn't. 
# We can fix it by not spliting the data, that we can at least create the databunch and view the data/train the model with default hyper-parameters
# 
# But you lose the ability to tune the model since you don't have a validation set, we will see how we can fix this problem later on. (At least we get the things going)

# In[ ]:


src = (ImageItemList.from_csv('../input/','train.csv',folder='train')
        .no_split()
        .label_from_df())


# In[ ]:


data = (src.transform(get_transforms(),size=224)
       .databunch()
       .normalize(imagenet_stats))


# In[ ]:


data.show_batch(rows=3,figsize=(12,7))


# Something went wrong again, as I researched a bit, this seems to be kaggle kernel / pytorch issue?
# But you can simply fix this problem by just let 1 single CPU handle the dataloading step

# In[ ]:


data = (src.transform(get_transforms(),size=224)
       .databunch(num_workers=0)
       .normalize(imagenet_stats))


# In[ ]:


data.show_batch(rows=3,figsize=(12,9))


# There we go, now the data is ready, we can start training the model.
# However, there is no way you can tune your model, you don't want to use test set to evaluate your model.
# 
# We need somehow give our model a validation set, so we know how well is our model.
# 
# **Split the data with 20% to validation set**
# 
# We failed spliting the data before, because we try to split the whole data as 80% train 20% validation.
# But if the class has only 1 image, we want it to be in the train-set. We will try our best to learn the class, and if we see it in the test set, we want to make right prediction. 
# 
# Therefore, we can do 20% split on the sub-set of the train set.
# 
# 1. If the class has more than 2 image, we pick 1 to the validation set
# 2. If the class has 2 or less image, we let it stay on the train set
# 
# Also, you can do data agrumentation. To create more images for each small classes to solve the issue (but you still want to make sure you sub-sampling correctly) 

# # **Sub-sampling**
# 
# 1. Create a copy of the df_label
# 2. Create a new column called total to track the number of images in that class
# 3. Create a new data frame that sub-sampling from each classes. 
#     * If the class has 1 image, we won't take it to validation set
#     * If the class has 2 images, we won't take it to validation set
#     * If the class has 3 images, we take 1 to validation set, and leave 2 in the train set
# and so on. 
# 
# We can also adjust the threshold later, for example, if we have 5 images, we take 1 to validation set, and leave the classes with 5 images or less in training. 

# In[ ]:


df_test_split = df_label.copy()
df_test_split['total'] = df_test_split.groupby('Id')['Id'].transform('count')
df_grouped = df_test_split.groupby('Id').apply(lambda x: x.sample(frac=0.2,random_state=47))
df_grouped.describe()


# Now we have randomly picked 4389 images from the train set, each of them have at least 3 images originally in the train set. 
# We can take a close look to make sure we are selecting the right ones

# In[ ]:


df_grouped.tail(10)


# In[ ]:


df_merged = pd.merge(left=df_test_split,right=df_grouped,on='Image',how='left',suffixes=('','_y'))
df_merged['is_valid'] = df_merged.Id_y.isnull()!=True
df_merged.head(20)


# Drop the merged colums that we dont need from the final table

# In[ ]:


df_merged.drop(['Id_y','total_y'],axis=1,inplace=True)
df_merged.head(10)


# Now we are ready, we have marked our original train-set with new columns if it should split to validation set or not. 
# We can start load it to FastAI 
# 
# Since  .from_csv will need a csv file, we will pack our dataframe to csv

# In[ ]:


df_merged.to_csv('validation_random.csv',index=False)


# In[ ]:


src = (ImageItemList.from_csv('../input/','/kaggle/working/validation_random.csv',folder='train')
        .split_from_df(col='is_valid')
        .label_from_df(cols='Id'))


# In[ ]:


data = (src.transform(get_transforms(max_zoom=1, max_warp=0),resize_method=ResizeMethod.SQUISH,size=224)
       .databunch(num_workers=0)
       .normalize(imagenet_stats))


# In[ ]:


data.show_batch(rows=3,figsize=(12,9))


# Now you have databunch object ready, and train-validation set ready. 
# You can start calling learner to build the baseline model.
# 
# Couple things note here:
# 1. You need to create a MAP5 metric to evaluate your model, FastAI doesn't have default MAP5 metric. 
# You can read more in [here](http:/https://www.kaggle.com/pestipeti/explanation-of-map5-scoring-metric/) to construct a MAP5 metric to evaluate your model
# 
# 2. Kaggle kernel doesn't allow you to download model to '../input/, so if you are just calling learner_create you will have some 'READ only' issue. (I havn't figured out a work around, so I trained on other VMs)
# 
# Thanks for taking your time to read this kernel. Hope this helps FastAI starters. 

# In[ ]:




