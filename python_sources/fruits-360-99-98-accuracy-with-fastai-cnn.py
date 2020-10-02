#!/usr/bin/env python
# coding: utf-8

# Hi, This is very first time I tried to build a deep learning model. if there are some error in my model please let me know
# I tried out fruit dataset with rasnet34(fastai) model

# This model will tell you about Fruit just by seeing there images. you can download the dataset that I used from kaggle dataset (https://www.kaggle.com/moltean/fruits). I have used a Convolutional neural network to build this model.
# 
# This project is a homework assignment for Fastai's Deep learning for coders lesson 1

# In[ ]:


from fastai.vision import *
from fastai.metrics import error_rate


# Current dataset directory is read-only; we might need to modify some training set data.
# Moving to /tmp gives us Write permission on the folder.

# In[ ]:


cp -r /kaggle/input/fruits-360_dataset/ /tmp


# In[ ]:


path= '/tmp/fruits-360_dataset/fruits-360/'


# Creating our databunch object. Fortunately this dataset uses imagenet style, so our factory method will suffice. We can experiment using different size values, I have found 224 to be optimal.
# 
# Fastai vision models are pretrained models on Imagenet. We are essentially using transfer learning here. So we should normalize our dateset using imagenet statistics as well.

# In[ ]:


data=ImageDataBunch.from_folder(path,train='Training',valid='Test',size=256,bs=64).normalize(imagenet_stats)
data.show_batch(rows=3,figsize=(8,8))


# Now let's create our learner. I'm using resnet34 for now with accuracy as metric.
# Resnet34 trains faster so let's go with that.

# In[ ]:


learner34=cnn_learner(data,models.resnet34,metrics=accuracy)


# In[ ]:


learner34.fit_one_cycle(4)


# 99% Accuracy using resnet34 is great. We might be able to increase our accuracy even more with resent50.
# Let's try that!

# In[ ]:


learner50=cnn_learner(data,models.resnet50,metrics=accuracy)


# Let's find our learning rate for resnet50 first. It's always a good idea to pass the learning rate while fitting.

# In[ ]:


learner50.lr_find()
learner50.recorder.plot()


# We have choose a maximum learning rate where the curve is most steep. So I've chosen 0.01 for this. Our CNN uses discriminative learning rate, so our max learning rate will be 0.01.

# In[ ]:


lr=1e-2


# In[ ]:


learner50.fit_one_cycle(4,slice(lr))


# Results are quite good.
# 
# Now let's see which cases we failed.

# In[ ]:


interp=ClassificationInterpretation.from_learner(learner50)
interp.plot_top_losses(9,figsize=(12,12))


# In[ ]:


interp.most_confused(min_val=1)


# In[ ]:


learner50.export("/kaggle/working/fruitsmodel.pkl")


# Thank you for sticking around.
