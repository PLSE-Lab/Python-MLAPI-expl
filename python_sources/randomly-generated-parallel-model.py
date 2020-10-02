#!/usr/bin/env python
# coding: utf-8

# # Random Ensembling as a Single Parallel Model
# 
# Hello, this is my first ever kernel in my first ever competition, so bear with me.
# I'm planning to make a simple fully-connected feed forward ensamble, randomly generated.
# To do this I'll be using the Keras Functional API with a tensorflow backend.
# 

# First, the imports. Nothing to non-standard:

# In[20]:


import random
import numpy as np

import pandas as pd

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Concatenate, BatchNormalization


# Here's the training data:

# In[21]:


import os
print(os.listdir("../input"))


# We need to be able to ensamble a large group of model-blocks

# In[22]:


def build_random_model_block(layer_inp, out_dim):
    depth = random.randint(1, 8)

    x = layer_inp
    layer = None
    for __ in range(depth):
        layer = random.choice(
             ["Dense"] # + 
#             (["Dropout"] if layer is not "Dropout" else []) +
#             (["BatchNormalization"] if layer is not "BatchNormalization" else [])
        )
    
        if layer == "Dense":
            units = 2 ** random.randint(1, 11)
            activation = random.choice(["tanh", "sigmoid", "relu", "selu", None])
            x = Dense(units, activation="relu")(x)
            
        if layer == "Dropout":
            dropout_rate = random.random()
            x = Dropout(dropout_rate)(x)
        
        if layer == "BatchNormalization":
            x = BatchNormalization()(x)
    
    out = Dense(out_dim, activation="sigmoid")(x)
        
    return out


# Now, we need to build a nice parallell model:

# In[23]:


def build_parallel_model(inp_shape, out_dim, models=16):
    inp = Input(shape=inp_shape)

    models = [build_random_model_block(inp, out_dim) for __ in range(models)]
    ensemble = Concatenate()(models)
    
    out = Dense(out_dim, activation="sigmoid")(ensemble)
    
    model = Model(inp, out)
    model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])
    
    return model


# training time!

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]


# In[ ]:


model = build_parallel_model((len(cols),), 1)
model.summary()


# In[ ]:


x = np.array([train[col] for col in cols]).T
print(x.shape)
y = np.array([[target] for target in train["target"]])
print(y.shape)
print(len(cols))
model.fit(x, y, batch_size=128, epochs=5)
print(model.evaluate(x, y, batch_size=128))


# In[ ]:


x_test = np.array([test[col] for col in cols]).T
print(x_test.shape)
print(len(cols))
predictions = model.predict(x_test, batch_size=128)

sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = predictions
sub.to_csv('submission.csv',index=False)

