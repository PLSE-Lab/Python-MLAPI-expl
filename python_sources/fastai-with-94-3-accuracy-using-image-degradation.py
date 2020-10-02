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


path = '../input/'


# In[ ]:


#lets use fastai to classfify these 6 classes of images
from fastai import *
from fastai.imports import *
from fastai.vision import *


# In[ ]:


#lets use databunch with data block api to create labels, classes and databunch


# In[ ]:


# note we will use Image Pixel Degradation trick to manipulate the resnet imagenet weights to 
# a more concise related to our data weights


# In[ ]:


# so here is the plan 
# we will first start with 64*64 and the we will increase it up to the maximum size
# as we train again and again


# In[ ]:


#sourcr
scr = ImageList.from_folder(path).split_by_folder(train  = 'seg_train',valid = 'seg_pred')


# In[ ]:


scr.label_from_folder().add_test_folder('seg_test')


# In[ ]:


src = ImageList.from_folder(path).split_by_folder(train = 'seg_train',valid = 'seg_test').label_from_folder().add_test_folder(test_folder = 'seg_valid')


# In[ ]:


data  = (src.transform(get_transforms(),size = 74).databunch().normalize(imagenet_stats))


# In[ ]:


data.show_batch(rows = 2, figsize = (12,9))


# In[ ]:


# so we are ready with the databunch we have created
# now we need to train the dataset and check the accuracy
# lets do it then


# In[ ]:


# creating the learner
learn = cnn_learner(data, models.resnet34,metrics = [error_rate,accuracy])


# In[ ]:


# now our learner is ready so lets use this to find a good learning rate
# to train the last layers of imagenet resnet model
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.model_dir  ='../kaggle/working'


# In[ ]:


#from the above graph we need to find the most steep path of the slope( or point)
# so from above we can clearly see that it is near the point 1e-03


# In[ ]:


#so lr
lr = 1e-03


# In[ ]:


# now lets train the last few fully connected layers of the resnet architecture
learn.fit_one_cycle(5,slice(lr))


# In[ ]:


# I think our model needs more epochs as it is still underfitting
# but for now lets move forward and train our data on fully layers in resnet architecture


# In[ ]:


learn.save('stage_1')


# In[ ]:


#learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(8,max_lr = slice(4e-05,lr/5))


# In[ ]:


# so yeah we got approx 93% which is not bad on reduced pixel sized images and 
# we will push this upto 95% by using the original sized image pixels


# In[ ]:


# but first lets analyze where our data is going wrong
learn.save('stage_2')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.most_confused(min_val = 2)


# In[ ]:


# so our data is mostly making mistakes on glacier as mountains 
# and vice versa 
# but we can fix this by providing more sized pixel


# In[ ]:


#plotting the confusion matrix
interp.plot_confusion_matrix()


# In[ ]:


# it's making mistakes more on street as buildings and mountains as glacier and vice versa for both


# In[ ]:


# now lets use the trick on our sleeves
#using original sized pixels
data = (src.transform(get_transforms(),size = 150).databunch().normalize(imagenet_stats))


# In[ ]:


learn.freeze()
learn.data = data


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(6,slice(6e-04))


# In[ ]:


learn.save('2_stage1')


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(6,max_lr = slice(1e-5,1e-03/10))


# In[ ]:


# so yeah we have got the accuracy close to 94.3% which is not bad we can further improve the accuracy by using resnet50 using the same method that we have used above


# In[ ]:


# lets interp this learnner now
learn.save('final_result_94.3%')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.most_confused()


# In[ ]:


# I think the galcier images are too confusing for our
# model and same goes for the buildings with streets
# the former one make sense as some glaciers are mountains
# as they are formed as a cover for the mountains at the higher
#elevation
# and same goes for the sreets with buildings as 
# some buildings are consists of more buildings so as to
# differentiate between them is quite a hard task


# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:


interp.plot_top_losses(k=2)


# In[ ]:


# as wee can see in the second image the glacier mountain is both the mountain as well 
# as glacier I think it should be on the seperate class of both galcier and mountains


# In[ ]:


path = '../input/seg_pred'


# In[ ]:


learn.export()


# In[ ]:


preds,y = learn.get_preds(path)


# In[ ]:


# lets check out preds on the valid data
from matplotlib import pyplot as plt


# In[ ]:


plt.imshow(preds[1:34])


# In[ ]:


y[1:34]


# In[ ]:





# In[ ]:


learn.model_dir ='../'


# In[ ]:


learn.save('yeah')


# In[ ]:





# In[ ]:


learn.save('yeah')


# In[ ]:




