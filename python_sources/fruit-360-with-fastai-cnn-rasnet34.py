#!/usr/bin/env python
# coding: utf-8

# Hi, This is very first time I tried to build a deep learning model. if there are some error in my model please let me know

# ## Fruit Classification Model
# This model will tell you about Fruit by seeing there images. you can download the dataset that I used from kaggle dataset (https://www.kaggle.com/moltean/fruits). I have used a Convolutional neural network to build this model.
# 
# This project is a homework assignment for Fastai's Deep learning for coders lesson 1

# In[1]:


from fastai.vision import *
from fastai.metrics import error_rate


# Current dataset directory is read-only; we might need to modify some training set data. Moving to /tmp gives us Write permission on the folder.(Thank to abyaadrafid for telling me how to do this)

# In[2]:


cp -r /kaggle/input/fruits-360_dataset/ /tmp


# In[3]:


path= '/tmp/fruits-360_dataset/fruits-360/'


# Fastai vision models are pretrained models on Imagenet. We are essentially using transfer learning here. So we should normalize our dateset using imagenet statistics as well.
# 
# 

# In[13]:


data=ImageDataBunch.from_folder(path,train='Training',valid='Test',size=256,bs=64).normalize(imagenet_stats)
data.show_batch(rows=3,figsize=(8,8))


# So Now we have created a databunch object, let's create our learner. I'm using resnet34 with accuracy as metric.

# In[6]:


learner = cnn_learner(data,models.resnet34,metrics=accuracy)


# Let's find our learning rate for resnet50 first. It's always a good idea to pass the learning rate while fitting.

# In[8]:


learner.lr_find()
learner.recorder.plot()


# We have choose a maximum learning rate where the curve is most steep. So I've chosen 1e-2 for this. 

# In[9]:


lr=1e-2


# In[11]:


learner.fit_one_cycle(4,slice(lr))


# Results are good! I got 99.73 accuracy
# It is always a good practice to see which cases we failed.

# In[15]:


interp=ClassificationInterpretation.from_learner(learner)
interp.plot_top_losses(9,figsize=(12,12))


# In[16]:


interp.most_confused(min_val=1)


# In[18]:


learner.export("/kaggle/working/fruits360model.pkl")


# Thank you for sticking around.
