#!/usr/bin/env python
# coding: utf-8

# Some trials on MNIST data comparing resnet18 and resnet 34 - trying out different learning rates

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from fastai.vision import *
from fastai.metrics import error_rate
from PIL import Image
from pathlib import Path


# In[ ]:


bs=64


# In[ ]:


folder_path = Path("../input/digit-recognizer")
os.listdir(folder_path)


# In[ ]:


folder_path.ls() #another (fast.ai way of looking in the folder)


# In[ ]:


#extract data from csv - using clunky names for initial extraction - will just copy so dont have to
#extract again if mess with data
train_data = pd.read_csv(folder_path/'train.csv')
test_data = pd.read_csv(folder_path/'test.csv')


# In[ ]:


train_data.tail()


# first column of each row is the label (dependent variable - or Y). 784 columns represent the independent (X)

# In[ ]:


test_data.tail(1)


# test data does not feature a label - as expected, we 

# In[ ]:


train_folder = Path("../train")
test_folder = Path("../test")


# In[ ]:


#create the training directory
for x in range(10):
    try:
        os.makedirs(train_folder/str(x))
    except:
        pass


# In[ ]:


os.listdir(train_folder)


# so now have 10 folders - one for each digit --> out of order bc folders made based on order encountered in training data

# In[ ]:


os.makedirs(test_folder)


# In[ ]:


#save training images into train_folder directory

for index, row in train_data.iterrows():
    label = row[0]
    digit = row[1:]
    
    filepath = train_folder/str(label)
    filename = f"{index}.jpg"
    
    digit = digit.values
    digit = digit.reshape(28, 28)
    digit = digit.astype(np.uint8)
    
    img = Image.fromarray(digit)
    img.save(filepath/filename)


# In[ ]:


# prints out the last go through for loop
print(filepath)
print(filename)
print(label)


# so the last item was the 42,000 one in training data. It was a 9 digit and saved in the 9 folder within the train directory. within the .jpg file the digit values were reshaped into a 28x28 matrix and cast to unsigned integer of 8 bits

# In[ ]:


#this saves the test images

for index, digit in test_data.iterrows():
    
    filepath = test_folder
    filename = f"{index}.jpg"
    
    digit = digit.values
    digit = digit.reshape(28,28)
    digit = digit.astype(np.uint8)
    
    img = Image.fromarray(digit)
    img.save(filepath/filename)


# In[ ]:


tfms = get_transforms(do_flip=False, max_zoom=1.2)


# In[ ]:


data = ImageDataBunch.from_folder(
    path = train_folder,
    test = test_folder,
    valid_pct = 0.2,
    bs = 32,
    size = 28,
    ds_tfms = tfms
)


# In[ ]:


data.normalize(mnist_stats)


# In[ ]:


data.classes


# In[ ]:


learn = cnn_learner(data, base_arch=models.resnet18, metrics=accuracy,
                   model_dir='/tmp/models', callback_fns=ShowGraph)


# This will run 5 epochs - keeping the original (pretrained) layers untouched and only messing with the two new ones that were added by fast.ai 

# In[ ]:


learn.fit_one_cycle(5)


# without doing any real work - just using the defaults in fast.ai and using pretrained weights, able to create model that is accurate 95.5% of time - 4.5% error rate! this rate is 2155 on leaderboard (out of 2450)

# In[ ]:


learn.save('defaults-frz-5ep')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_top_losses(9, figsize=(7,7))


# some of these even I would have trouble with!

# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:


interp.most_confused(10) #shows the ones with at least 10 confusions


# let's unfreeze and run the model for a few epochs - see how it goes, then bring back the weights and run it with a specific learning rate

# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(5) #


# After 5 epochs seeing 99.4% accuracy! up from 95.5% (without unfreezing!)
# This accuracy would place you at approx 580 on leaderboard, so around just into the top 25%!!!
# 
# Now let's load up the old weights and go through a few different learning rates (with the same number of epochs) and see how it would impact the accuracy

# In[ ]:


learn.load('defaults-frz-5ep')
learn.unfreeze()
learn.fit_one_cycle(5, max_lr=slice(1e-5, 1e-3))


# In[ ]:


#lets invert it for one and see how badly it does
learn.load('defaults-frz-5ep')
learn.unfreeze()
learn.fit_one_cycle(5, max_lr=slice(1e-3, 1e-5))


# looks like having it start higher and drop was the way to go - lets try to go a few more epochs or jack it up a little higher

# In[ ]:


learn.load('defaults-frz-5ep')
learn.unfreeze()
learn.fit_one_cycle(10, max_lr=slice(1e-3, 1e-5))


# In[ ]:


learn.load('defaults-frz-5ep')
learn.unfreeze()
learn.fit_one_cycle(10, max_lr=slice(3e-3, 3e-5))


# In[ ]:


learn.load('defaults-frz-5ep')
learn.unfreeze()
learn.fit_one_cycle(10, max_lr=slice(3e-3, 3e-2))


# Now let's change the resnet arch

# In[ ]:


learn = cnn_learner(data, base_arch=models.resnet34, metrics=accuracy,
                   model_dir='/tmp/models', callback_fns=ShowGraph)


# In[ ]:


learn.fit_one_cycle(5)


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.save('frzn-res34')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(5)


# In[ ]:


learn.fit_one_cycle(5)


# try again after reloading data - and see if this changes (shouldn't have)

# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(5)


# No - reloading data was not needed. lol

# In[ ]:


learn.load('frzn-res34')
learn.unfreeze()
learn.fit_one_cycle(5, max_lr=slice(1e-6, 1e-4))


# In[ ]:


learn.load('frzn-res34')
learn.unfreeze()
learn.fit_one_cycle(5, max_lr=slice(1e-5, 1e-3))


# In[ ]:


learn.load('frzn-res34')
learn.unfreeze()
learn.fit_one_cycle(5, max_lr=slice(1e-3, 1e-5))


# In[ ]:


learn.load('frzn-res34')
learn.unfreeze()
learn.fit_one_cycle(5, max_lr=slice(1e-2, 1e-4))


# In[ ]:


learn.load('frzn-res34')
learn.unfreeze()
learn.fit_one_cycle(5, max_lr=slice(1e-3, 1e-2))


# In[ ]:


learn.load('frzn-res34')
learn.unfreeze()
learn.fit_one_cycle(5, max_lr=slice(3e-2, 3e-3))


# nothing beat the defaults! let's run the default at 10 epochs and see how it looks

# In[ ]:


learn.load('frzn-res34')
learn.unfreeze()
learn.fit_one_cycle(10)


# why is this so eratic - even in first 5 epochs - you saw it kind of flying around this time vs the one earlier.
# 
# The original try with resnet34 - for 5 epochs after unfreezing (initial run there was 5 epochs too) got 99.58% accuracy. This used the default for learning rate --> this would put you at 380 on kaggle (out of 2450) - approx. top 15.5%

# So in next set - try using resnet34 - do 5 epochs frozen - do 5 epochs unfrozen
# *try out different momentums
# *find another variable to mess around with as well to test
# *maybe play around with transformations?
# 

# extension to try out res34 & res50

# In[ ]:


learn = cnn_learner(data, base_arch=models.resnet34, metrics=accuracy,
                   model_dir='/tmp/models', callback_fns=ShowGraph)


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(3)


# In[ ]:


learn.save('res34-froze-3ep')


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(10, max_lr=slice(3e-3, 3e-1))


# maybe needed to run a few more epochs before unfreezing

# In[ ]:


learn = cnn_learner(data, base_arch=models.resnet34, metrics=accuracy,
                   model_dir='/tmp/models', callback_fns=ShowGraph)


# In[ ]:


learn.fit_one_cycle(5)


# In[ ]:


learn.save('res34-froze-5ep')


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(10)


# In[ ]:


learn.load('res34-froze-5ep')
learn.unfreeze()
learn.fit_one_cycle(10, max_lr=slice(3e-3,3e-2))


# In[ ]:


learn.load('res34-froze-5ep')
learn.unfreeze()
learn.fit_one_cycle(10, max_lr=slice(3e-2,3e-3))


# 

# In[ ]:


learn = cnn_learner(data, base_arch=models.resnet50, metrics=accuracy,
                   model_dir='/tmp/models', callback_fns=ShowGraph)


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5)


# In[ ]:


learn.save('res50-froze-5ep')


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(10, max_lr=slice(3e-3, 3e-2))


# seems like the defaults are good enough and messing with the LR for MNIST is not a good idea!

# 
