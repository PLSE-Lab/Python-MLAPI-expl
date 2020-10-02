#!/usr/bin/env python
# coding: utf-8

# In[ ]:


ls -la ../input/keras-pretrained-models


# In[ ]:


import os
import shutil
print(os.listdir("../input"))


# In[ ]:


os.makedirs('/tmp/.keras/datasets')


# In[ ]:


shutil.copytree("../input/keras-pretrained-models", "/tmp/.keras/models")


# In[ ]:


print(os.listdir("/tmp/.keras/models"))


# In[ ]:





# In[ ]:


from keras import applications
model = applications.resnet50.ResNet50(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000)


# In[ ]:


model.summary()

