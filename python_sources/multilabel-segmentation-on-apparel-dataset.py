#!/usr/bin/env python
# coding: utf-8

# # **Multilabel classification using Fastai-v1**

# This is an exercise kernel for the fastai lesson 3. I have also visited the "Multi-label Classification using FastAi Library" by Kais K in kaggle. You can visit his kernel [*here*](http://www.kaggle.com/kaiska/multi-label-classification-using-fastai-library)
# 
# 
# I will explain my steps in small sentences for you to get a basic understanding of what am I trying to accomplish, but if you want a detailed explanation of the functions used here, [fastai docs](http://docs.fast.ai/) is the best place to learn.

# ### Getting started
# 
# This is the kaggle default cell, which helps in listing directories where data is present. Then the necessary packages are downloaded.

# In[ ]:




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


from fastai import *
from fastai.vision import *


# ### Initiating the path variable
# 
# 

# In[ ]:


path = Path('/kaggle/input/apparel-dataset/')
path.ls()


# An image list is created to read all the images from the path folder by folder. The function "from_folder" is important as the labels of the images are the folder names, the labels are created from the name ofthe folder in which the image is in(in the next few steps)

# In[ ]:


img = ImageList.from_folder(path, recurse = True)


# To see how many files are present in the image list

# In[ ]:


img.items.shape


# In[ ]:


img.open(img.items[10])


# ### Creating Training and Validation folders 
# 
# Here, a training and validation folders are created from the imagelist that we previously created. The labels of the images are taken from the folder name, with a delimiter **'_'**. 
# 
# random seed is set to a fixed number so that we will always get the same training and validation split thereby reproducing the results.

# In[ ]:


np.random.seed(33)
src = (img.split_by_rand_pct(0.2).label_from_folder(label_delim = '_'))


# ### Databunch created
# 
# In fastai library, the training neuralnetwork takes in a object called databunch, which consists of training and validation sets (optionally test set). This databunch will receive images and labels with data augmentations done using the **transform** class. The  images are normalized to imagenet stats as we are using a pretrained imagenet neural network.

# In[ ]:


tfms = get_transforms()
data = (src.transform(tfms, size =128).databunch().normalize(imagenet_stats))


# In[ ]:


data.show_batch(rows = 3, figsize = (15,11))
print(f""" list of classes in the dataset: {data.classes}\n
        number of labels in the dataset: {data.c}\n
        length of training data: {len(data.train_ds)}\n
        length of validation data: {len(data.valid_ds)}""")


# ### Creating custom metrics and a learner
# 
# Here, we created a custom metric from the accuracy metric as we are dealing with multi-label Classification.
# 
# Then a learner is created, which takes the databunch created above and downloads a pretrained model to train that data on.

# In[ ]:


acc_02 = partial(accuracy_thresh, thresh = 0.2)
learn = cnn_learner(data, models.resnet34, metrics = acc_02)


# In[ ]:


learn.model_dir = '/kaggle/working/models'


# ### Finding learning rate
# 
# **lr_find()** helps the learner to get the best learning rate for that model with the available data. This function will decrease the number of experimentations and thereby avoids the possibilty of overfitting.

# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# We can see that the steepest decrease in the loss is at 1e-2. So, taking this learning rate would be an ideal place to start training the model.

# In[ ]:


lr = 1e-2
learn.fit_one_cycle(5,slice(lr))


# In[ ]:


learn.save('stage-1-128')


# ### Training the model a little further
# 
# Though we got a good model with just 5 epochs, this learner is using the pretrained weights of resnet34 which trained on imagenet. To customize this learner for this data, we unfreeze the learner and train the later parts of the learner more to get even more accurate results.

# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# Here, the loss is incresing after 1e-3. So, the ideal learning rate will be 10 times lowerthan that. Since we are interested in training the later parts of the deep neural network more, we can slice the learning rate.

# In[ ]:


learn.fit_one_cycle(2, max_lr = slice(3e-6, 3e-4))


# In[ ]:


learn.recorder.plot_losses()


# The accuracy may further increase but may also resultin overfitting, so I have stopped training the learner.

# In[ ]:


learn.save('stage-2-128')


# In[ ]:




