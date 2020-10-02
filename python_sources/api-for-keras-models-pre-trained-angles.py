#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# There are some amazing baseline kernels shared with all kinds of models (either custom or pre-trained) throughout this competition. However, things can get messy really fast if we just keep piling on various snippets into our own codebase - different custom and Keras pre-trained models examples, some people allow base model training and some don't, some people use angles and some don't, and so on. 
# 
# Below I propose a clean API to remedy these issues by combining Factory Pattern with Python's lambas.

# In[ ]:


import numpy as np, pandas as pd
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Input, concatenate, GlobalMaxPooling2D

# only needed for Kaggle Kernel, locally just use weights='imagenet'
vgg16_fl = "../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"


# ## Basic setup
# 
# Nothing exciting just yet, here we import pre-trained models we might like to use and additionally define a simple ConvNet model to show how to use custom models with API below. Notice that we only define base conv layers for our custom model.

# In[ ]:


from keras.applications import VGG16, VGG19, ResNet50, Xception

def get_simple(dropout=0.5):
    model = Sequential()

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(75, 75, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(dropout))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(dropout))

    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(dropout))

    return model


# ## API
# 
# And here's where all the magic happens. Thanks to Python lambdas, we can keep `get_model` clean & small in size, but abstract enough to support use of any base model, adding angle as input, turning base training on/off, and customizing FC layers. Note that the aim was to keep it simple rather than full of features. Some things you might want to add: 
# 
# * Define separate dropout values per FC layer
# * Fine-tune pre-trained model by training only on a sub-set of layers (instead of current on/off switch)
# * Use `Flatten` or `GlobalAveragePooling2D` instead of `GlobalMaxPooling2D`

# In[ ]:


factory = {
    'vgg16': lambda: VGG16(include_top=False, input_shape=(75, 75, 3), weights=vgg16_fl),
    'vgg19': lambda: VGG19(include_top=False, input_shape=(75, 75, 3)),
    'xception': lambda: Xception(include_top=False, input_shape=(75, 75, 3)),
    'resnet50': lambda: ResNet50(include_top=False, input_shape=(200, 200, 3)),
    'simple': lambda: get_simple()
}

def get_model(name='simple',train_base=True,use_angle=False,dropout=0.5,layers=(512,256)):
    base = factory[name]()
    inputs = [base.input]
    x = GlobalMaxPooling2D()(base.output)

    if use_angle:
        angle_in = Input(shape=(1,))
        angle_x = Dense(1, activation='relu')(angle_in)
        inputs.append(angle_in)
        x = concatenate([x, angle_x])

    for l_sz in layers:
        x = Dense(l_sz, activation='relu')(x)
        x = Dropout(dropout)(x)

    x = Dense(1, activation='sigmoid')(x)

    for l in base.layers:
        l.trainable = train_base

    return Model(inputs=inputs, outputs=x)


# ## Example
# 
# Let's try it out!  
# Note that this is only for illustrative purposes, in practice you'd need to run this on GPU and preferably in two stages where you first train your FC layers and then fine-tune the rest of the model.

# In[ ]:


data = pd.read_json('../input/statoil-iceberg-classifier-challenge/train.json')
b1 = np.array(data["band_1"].values.tolist()).reshape(-1, 75, 75, 1)
b2 = np.array(data["band_2"].values.tolist()).reshape(-1, 75, 75, 1)
b3 = b1 + b2

X = np.concatenate([b1, b2, b3], axis=3)
y = np.array(data['is_iceberg'])
angle = np.array(pd.to_numeric(data['inc_angle'], errors='coerce').fillna(0))


# In[ ]:


model = get_model('vgg16', train_base=False, use_angle=True)
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])
history = model.fit([X, angle], y, shuffle=True, verbose=1, epochs=5)

