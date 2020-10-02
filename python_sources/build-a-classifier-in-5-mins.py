#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.applications.vgg19 import VGG19
from keras.layers import Flatten, Dense, Dropout
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


# create an image generator
data_gen = ImageDataGenerator(rescale = 1./255, validation_split = 0.2)


# In[ ]:


batch_size = 16

train_gen = data_gen.flow_from_directory("../input/flowers-recognition/flowers/flowers", target_size = (224, 224),
                                              batch_size = batch_size, class_mode = "categorical",
                                              subset = "training")
valid_gen = data_gen.flow_from_directory("../input/flowers-recognition/flowers/flowers", target_size = (224, 224),
                                             batch_size = batch_size, class_mode = "categorical",
                                             subset = "validation")


# In[ ]:


# create the base pre-trained model
base_model = VGG19(input_shape = (224, 224, 3), include_top = False, weights = None) 
base_model.load_weights("../input/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5")
for layer in base_model.layers:
    layer.trainable = False

X = base_model.output
X = GlobalAveragePooling2D()(X)
X = Dense(128, activation = "relu")(X)
X = Dropout(0.5)(X)
X = Dense(32, activation = "relu")(X)
predictions = Dense(5, activation = "softmax")(X)

model = Model(inputs = base_model.input, outputs = predictions)


# In[ ]:


model.summary()


# In[ ]:


model.compile(loss = "categorical_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"]
             )


# In[ ]:


train_history = model.fit_generator(train_gen, steps_per_epoch = len(train_gen), epochs = 10)


# In[ ]:


model.evaluate_generator(valid_gen, steps = len(valid_gen))


# In[ ]:


plt.subplot(1, 2, 1)
plt.plot(train_history.history["acc"])
plt.title("Accuracy")
plt.xlabel("epoch")

plt.subplot(1, 2, 2)
plt.plot(train_history.history["loss"])
plt.title("Loss")
plt.xlabel("epoch")


# In[ ]:




