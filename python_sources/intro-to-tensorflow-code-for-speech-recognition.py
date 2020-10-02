#!/usr/bin/env python
# coding: utf-8

# # Intro to Tensorflow 
# 
# Before engaging with this dataset I would recommend reading through https://www.kaggle.com/davids1992/data-visualization-and-investigation 

# ## Let's start by Identifying our knowns...
# 
# There are only 12 possible labels for the Test set: `yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`, `silence`, `unknown`.
# 
# The unknown label should be used for a command that is not one one of the first 10 labels or that is not silence.

# In[10]:


POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown'.split()
AUDIO_PATH = '../input/train/audio/'
AUDIO_PATHS = {}
for label in POSSIBLE_LABELS:
    AUDIO_PATHS[label] = AUDIO_PATH + label
print(AUDIO_PATHS)


# ## First we need to turn our audio files into numbers 
# then we can throw them into tensorflow 

# In[ ]:




