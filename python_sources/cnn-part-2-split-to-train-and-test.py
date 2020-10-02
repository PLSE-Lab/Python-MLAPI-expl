#!/usr/bin/env python
# coding: utf-8

# This is the 2nd part of my mini series in Detecting Respiratory Disease using Respiratory Audio (breathing sounds). Here are the other parts in this series:
# - Part 1: [Slicing audio files to subslices as defined by the .txt files](https://www.kaggle.com/danaelisanicolas/cnn-part-1-create-subslices-for-each-sound)
# - Part 3: [Creating Spectogram images from sound files](https://www.kaggle.com/danaelisanicolas/cnn-part-3-create-spectrogram-images)
# - Part 4: [Creating a model and training using VGG16](https://www.kaggle.com/danaelisanicolas/cnn-part-4-training-and-modelling-with-vgg16)
# 
# For this kernel, we'll just split the output from part 1 (wav slices) to train and validation sets. I'm importing my output from the Part 1 kernel which you can checkout from the links above.

# In[ ]:


get_ipython().system("ls '../input/cnn-part-1-create-subslices-for-each-sound/output/'")


# This task involves splitting without distorting the folder structure. I'm using split_folders python package to do this. Source: https://pypi.org/project/split-folders/

# In[ ]:


get_ipython().system('pip install split_folders')


# In[ ]:


import split_folders

import os


# In[ ]:


os.makedirs('output')
os.makedirs('output/train')
os.makedirs('output/val')


# In[ ]:


audio_loc = '../input/cnn-part-1-create-subslices-for-each-sound/output/'

split_folders.ratio(audio_loc, output='output', seed=1337, ratio=(0.8, 0.2))


# And that's that. You can now use the output from this kernel for Part 3 (WIP)
