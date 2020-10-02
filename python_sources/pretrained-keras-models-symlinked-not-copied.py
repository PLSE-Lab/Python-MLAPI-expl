#!/usr/bin/env python
# coding: utf-8

# The [Keras Pretrained Models dataset](https://www.kaggle.com/gaborfodor/keras-pretrained-models) is an awesome dataset which provides us a fast way to get into Kaggle's computer vision challenges without having to train models ourselves.
# 
# The suggested [way](https://www.kaggle.com/gaborfodor/resnet50-example/) to make use of this is to copy the weights in this dataset to `.keras/models`. The problem with this approach is it consumes too much disk space (the entire dataset is 943 MB, even copying just one model will consume at least 100MB) and Kaggle Kernels only have 1 GB to play with.
# 
# This notebook demonstrates how to use a [symlink](https://en.wikipedia.org/wiki/Symbolic_link) instead which works without using up a lot of resources.

# In[1]:


import matplotlib.pyplot as plt
import numpy             as np
import os

from keras.preprocessing               import image
from keras.applications.resnet50       import ResNet50, preprocess_input
from keras.applications.imagenet_utils import decode_predictions


# ## Create ~/.keras/models

# In[2]:


inputs_dir = "/kaggle/input"
models_dir = os.path.expanduser(os.path.join("~", ".keras", "models"))
os.makedirs(models_dir)


# ## Create the symlinks

# In[3]:


for file in os.listdir(inputs_dir):
    if file.endswith(".json") or file.endswith(".h5"):
        os.symlink(
            os.path.join(inputs_dir, file),
            os.path.join(models_dir, file)
        )


# In[4]:


get_ipython().system('ls  ~/.keras/models')


# ## Read example image

# In[7]:


fig, ax = plt.subplots(1, figsize=(12, 10))
img = image.load_img('../input/Kuszma.JPG')
img = image.img_to_array(img)
ax.imshow(img / 255.) 
ax.axis('off')
plt.show()


# ## Use model with pretrained weights

# In[8]:


resnet = ResNet50()


# In[9]:


img = image.load_img('../input/Kuszma.JPG', target_size=(224, 224))
img = image.img_to_array(img)
x = preprocess_input(np.expand_dims(img.copy(), axis=0))
preds = resnet.predict(x)
decode_predictions(preds, top=5)


# We get the same results as the reference notebook.
