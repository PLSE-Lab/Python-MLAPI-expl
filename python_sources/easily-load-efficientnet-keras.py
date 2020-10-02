#!/usr/bin/env python
# coding: utf-8

# # Installing Efficientnet
# 
# First, we will need to install efficientnet using pip. This is possible since we are importing the github source code as a dataset, and with Pip it is possible to install a library directly from disk. This means we do not need internet connection for this to work, and it works with GPU.

# First let's verify if `setup.py` is inside the github project:

# In[ ]:


get_ipython().system('ls ../input/efficientnet-keras-source-code/repository/qubvel-efficientnet-c993591/ ')


# Once that is confirmed, we can install it using pip:

# In[ ]:


get_ipython().system('pip install ../input/efficientnet-keras-source-code/repository/qubvel-efficientnet-c993591/')


# In[ ]:


get_ipython().system('pip show efficientnet')


# # Usage
# 
# Below, we show how to load efficientnet using Keras. Note that we can't use the util `keras.models.load_model`, since it will throw errors.

# In[ ]:


import efficientnet.keras as efn

model = efn.EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
model.load_weights('../input/efficientnet-keras-weights-b0b5/efficientnet-b0_imagenet_1000_notop.h5')

model.summary()

