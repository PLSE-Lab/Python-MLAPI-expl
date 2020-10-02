#!/usr/bin/env python
# coding: utf-8

# # InceptionV3 with varying input shapes
# 
# This is to show that the inception architecture is a procedure. Notice that the output shape is relative to the input shape. At the end, I've exploded the entire network layer list, so you can see the inception procedure handling input shapes of varying size.
# 
# **Be not confused!** Once a network is made, its shape is forever locked. You can only define shape when a network is created. However, you can *make* the network with whatever shapes you want (so long as they're compatible with the sequential operations of the network).

# In[ ]:


from tensorflow.keras import backend as K
from tensorflow.keras import layers as L
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications.inception_v3 import InceptionV3


# ## Input shape: (299,299,3)

# In[ ]:


K.clear_session()

input_tensor = L.Input(shape=(299,299,3))
inception = InceptionV3(include_top=False, # leaves out the classifier end
                        input_shape=(299,299,3),
                        pooling=None) # leaves out flattening/global pooling at end
output = inception(input_tensor)
model = Model(input_tensor,output) # as of this moment, this network's shapes are locked.
model.summary()


# ## Input shape: (96,96,3)

# In[ ]:


K.clear_session()

input_tensor = L.Input(shape=(96,96,3))
inception = InceptionV3(include_top=False,
                        input_shape=(96,96,3),
                        pooling=None)
output = inception(input_tensor)
model = Model(input_tensor,output)
model.summary()


# ## Input shape: (123,666,3)
# I'm only doing 3 channel inputs because Keras Applications only supports 3 channel input. If you code the inception network yourself, you can make it whatever you please.

# In[ ]:


K.clear_session()

input_tensor = L.Input(shape=(123,666,3))
inception = InceptionV3(include_top=False,
                        input_shape=(123,666,3),
                        pooling=None)
output = inception(input_tensor)
model = Model(input_tensor,output)
model.summary()


# ## Fully exploded examples:
# between these next two examples, look at the outputs. The layer-by-layer shapes are different numerically, but the same operations have been performed on them sequentially.
# 
# It will look a bit confusing, since inception networks have concatenations and whatnot going on. So, first, have a look at this visualization, so the linear format (`inception.summary()`) isn't so confusing.

# In[ ]:


K.clear_session()

inception = InceptionV3(include_top=False,
                        input_shape=(299,299,3),
                        pooling=None)

plot_model(inception)


# In[ ]:


inception.summary()


# different input shape

# In[ ]:


K.clear_session()

inception = InceptionV3(include_top=False,
                        input_shape=(256,256,3),
                        pooling=None)
inception.summary()


# ## Ok, now what?
# You already know this part, but here it is anyway, for a complete mental picture.

# In[ ]:


K.clear_session()

input_tensor = L.Input(shape=(299,299,3))
inception = InceptionV3(include_top=False,
                        input_shape=(299,299,3),
                        pooling=None) # or make it 'avg' or 'max' and remove global pooling layer
x = inception(input_tensor)
x = L.GlobalMaxPooling2D()(x)
x = L.Dense(1024,activation='relu')(x)
x = L.Dense(69,activation='softmax',name='predictions_yayyy')(x)

model = Model(input_tensor,x)

display(model.summary())
display(plot_model(model))

