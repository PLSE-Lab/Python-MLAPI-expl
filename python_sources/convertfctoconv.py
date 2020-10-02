#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.applications import VGG16
from keras import backend as K
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Input


# In[ ]:


K.clear_session()


# In[ ]:


model = VGG16(weights='imagenet')


# In[ ]:


model.summary()


# In[ ]:


model.get_layer('fc1')


# In[ ]:


model.layers.pop()


# In[ ]:


model.summary()


# In[ ]:


model.layers.pop()


# In[ ]:


model.summary()


# In[ ]:


model.layers.pop()
model.layers.pop()


# In[ ]:


model.summary()


# In[ ]:


model.layers[-1].output


# In[ ]:


fc2conv1 = Conv2D(4096, kernel_size=[7,7], strides=(1,1), padding='valid', activation='relu')(model.layers[-1].output)


# In[ ]:


fc2conv2 = Conv2D(4096, kernel_size=[1,1], strides=(1,1), padding='valid', activation='relu')(fc2conv1)


# In[ ]:


fc2conv3 = Conv2D(1000, kernel_size=[1,1], strides=(1,1), padding='valid', activation='relu')(fc2conv2)


# In[ ]:


fcnModel = Model(inputs=model.input, outputs=fc2conv3)


# In[ ]:


fcnModel.summary()


# In[ ]:




