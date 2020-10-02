#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# In[ ]:


from tensorflow.data import Dataset

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, MaxPool2D, Dense
from tensorflow.keras.activations import relu
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy, sparse_categorical_accuracy, SparseCategoricalAccuracy, CategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ## networks

# In[ ]:


def get_shallow_cnn():
  model = Sequential()
  model.add(Conv2D(16, 3, activation=relu, input_shape=(28, 28, 1)))
  model.add(MaxPool2D())
  model.add(Conv2D(32, 3, activation=relu))
  model.add(MaxPool2D())
  model.add(Flatten())
  model.add(Dense(128, activation=relu))
  model.add(Dense(10))
  return model


# ## mnist

# ### Dataset
# 
# In this section we will use the tensorflow Dataset object to create a generator for train, test, valid data and use it it to train the network.

# In[ ]:


(x_data, y_data), (x_test, y_test) = mnist.load_data()
x_data = x_data[..., np.newaxis]
x_test = x_test[..., np.newaxis]


# In[ ]:


x_data.shape, y_data.shape, x_test.shape, y_test.shape


# In[ ]:


(x_train, x_valid, y_train, y_valid) = train_test_split(
    x_data, y_data, test_size=0.15, random_state=42)


# In[ ]:


print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)


# In[ ]:


train_dataset = Dataset.from_tensor_slices((x_train, y_train))
valid_dataset = Dataset.from_tensor_slices((x_valid, y_valid))
test_dataset = Dataset.from_tensor_slices((x_test, y_test))


# In[ ]:


train_dataset = train_dataset.batch(64, True)
valid_dataset = valid_dataset.batch(64, True)
test_dataset = test_dataset.batch(64, True)


# In[ ]:


steps_per_epoch = x_train.shape[0]//64
validation_steps = x_valid.shape[0]//64


# #### using model compiled with 'accuracy':

# In[ ]:


model1 = get_shallow_cnn()

model1.compile(optimizer=Adam(),
              loss=SparseCategoricalCrossentropy(True),
              metrics=['accuracy'])

history1 = model1.fit(train_dataset, steps_per_epoch=steps_per_epoch,
              validation_data=valid_dataset,
              epochs=10, validation_steps=validation_steps)


# #### using model compiled with 'sparse categorical accuracy':

# In[ ]:


model2 = get_shallow_cnn()

model2.compile(optimizer=Adam(),
              loss=SparseCategoricalCrossentropy(True),
              metrics=[SparseCategoricalAccuracy()])

history2 = model2.fit(
    train_dataset, steps_per_epoch=steps_per_epoch,
    validation_data=valid_dataset,
    epochs=10, validation_steps=validation_steps)


# #### using model compiled with 'Accuracy':

# 
# 
# ```
# model3 = get_shallow_cnn()
# 
# model3.compile(optimizer=Adam(),
#               loss=SparseCategoricalCrossentropy(True),
#               metrics=[Accuracy()])
# 
# history3 = model3.fit(
#     train_dataset, steps_per_epoch=steps_per_epoch,
#     validation_data=valid_dataset,
#     epochs=10, validation_steps=validation_steps)
# 
# # Results in an error:
#     ValueError: Shapes (64, 10) and (64, 1) are incompatible
# 
# ```
# 
# 

# In[ ]:


y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)
y_test = to_categorical(y_test)
print(y_train.shape, y_valid.shape, y_test.shape)
train_dataset3 = Dataset.from_tensor_slices((x_train, y_train))
valid_dataset3 = Dataset.from_tensor_slices((x_valid, y_valid))
test_dataset3 = Dataset.from_tensor_slices((x_test, y_test))


# In[ ]:


train_dataset3.element_spec


# In[ ]:


train_dataset3 = train_dataset3.batch(64, True)
valid_dataset3 = valid_dataset3.batch(64, True)
test_dataset3 = test_dataset3.batch(64, True)


# In[ ]:


train_dataset3.element_spec


# In[ ]:


model3 = get_shallow_cnn()

model3.compile(optimizer=Adam(),
              loss=CategoricalCrossentropy(True),
              metrics=[CategoricalAccuracy()])

history3 = model3.fit(
    train_dataset3, steps_per_epoch=steps_per_epoch,
    validation_data=valid_dataset3,
    epochs=10, validation_steps=validation_steps)


# In[ ]:


history2.history.keys()


# In[ ]:


plt.style.use('ggplot')
fig, axes = plt.subplots(1, 2, sharex=True, figsize=(12, 5))

axes[0].set_xlabel("Epochs", fontsize=14)
axes[0].set_ylabel("Loss", fontsize=14)
axes[0].set_title('Loss vs epochs')
axes[0].plot(history1.history["loss"])
axes[0].plot(history2.history["loss"])
axes[0].plot(history3.history["loss"])

axes[1].set_title('Accuracy vs epochs')
axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epochs", fontsize=14)
axes[1].plot(history1.history["accuracy"])
axes[1].plot(history2.history["sparse_categorical_accuracy"])
axes[1].plot(history3.history["categorical_accuracy"])

plt.show()


# Why are they exactly not the same??
# 
# From which places did the randomness creep in.

# ### ImageDataGenerator
# 
# tf.keras image data generator object is specifically designed for images

# In[ ]:


(x_data, y_data), (x_test, y_test) = mnist.load_data()
x_data = x_data[..., np.newaxis]
x_test = x_test[..., np.newaxis]


# In[ ]:


x_data.shape, y_data.shape, x_test.shape, y_test.shape


# In[ ]:


(x_train, x_valid, y_train, y_valid) = train_test_split(
    x_data, y_data, test_size=0.15, random_state=42)


# In[ ]:


img_generator = ImageDataGenerator(rescale=1/255.0)


# In[ ]:


train_dataset = img_generator.flow(x_train, y_train, batch_size=64)
valid_dataset = img_generator.flow(x_valid, y_valid, batch_size=64)
test_dataset = img_generator.flow(x_test, y_test, batch_size=64)


# In[ ]:


model5 = get_shallow_cnn()

model5.compile(optimizer=Adam(),
              loss=SparseCategoricalCrossentropy(True),
              metrics=['accuracy'])

history5 = model5.fit(train_dataset, steps_per_epoch=steps_per_epoch,
              validation_data=valid_dataset,
              epochs=10, validation_steps=validation_steps)


# In[ ]:


plt.style.use('ggplot')
fig, axes = plt.subplots(1, 2, sharex=True, figsize=(12, 5))

axes[0].set_xlabel("Epochs", fontsize=14)
axes[0].set_ylabel("Loss", fontsize=14)
axes[0].set_title('Loss vs epochs')
axes[0].plot(history5.history["loss"])

axes[1].set_title('Accuracy vs epochs')
axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epochs", fontsize=14)
axes[1].plot(history5.history["accuracy"])

plt.show()


# In[ ]:




