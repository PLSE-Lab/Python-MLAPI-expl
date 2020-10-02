#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from fastai.vision import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


# ### Preparing the data

# In[ ]:


#Check the folder where images are kept
path = Path('/kaggle/input/flowers-recognition/flowers/flowers/')
path.ls()


# In[ ]:


#Set the 5 output classes
classes = ['sunflower','tulip','rose','dandelion','daisy']


# In[ ]:


#Preparing the data bunch object which holds the image data
#define the batch size
bs = 16

#lets do some data augmentation
#tfms = get_transforms(do_flip=False)

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=224, bs=bs, num_workers=4).normalize(imagenet_stats)


# ### View the data
# 

# In[ ]:


data.show_batch(rows=3, figsize=(7,8))


# In[ ]:


#lets recheck the classes
data.classes


# In[ ]:


#lets check the stats
data.classes, data.c, len(data.train_ds), len(data.valid_ds)


# ### Lets train the Resnet50 Model

# In[ ]:


#define the model 
#Make sure you have the internet switch in kaggle on to download the model
learn = cnn_learner(data, models.resnet50, metrics=[error_rate, accuracy], model_dir='/output/model/')


# In[ ]:


#Lets find the correct range for learning rate
learn.lr_find()
learn.recorder.plot()


# In[ ]:


#Lets train the model for 8 cycles
learn.fit_one_cycle(8)


# In[ ]:


#saving the model 
learn.save('stage-1-50')


# In[ ]:


learn.unfreeze()


# In[ ]:


#Lets see if we can fine tune it a bit
learn.fit_one_cycle(3, max_lr=slice(1e-5,1e-3))


# In[ ]:


learn.save('stage-2-50')

