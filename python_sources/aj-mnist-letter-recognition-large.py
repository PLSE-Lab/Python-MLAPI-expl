#!/usr/bin/env python
# coding: utf-8

# # Intro
# 
# * Hello
# * I think this is the simplest way to deal with the large dataset, regarding of the test data loading or the model itself

# # Import

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

import os
import cv2
import random

import matplotlib.pyplot as plt


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split


# # Loading Test Data

# In[ ]:


Datadir = '../input/notMNIST_small/notMNIST_small'
Categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

img_size = 50
training_data = []

for Category in Categories:
    path = os.path.join(Datadir, Category)
    print("Loading data from ", path)
    class_num = Categories.index(Category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(
                path, img), cv2.IMREAD_GRAYSCALE)
            img_resize = cv2.resize(img_array, (img_size, img_size))
            training_data.append([img_resize, class_num])
        except Exception as e:
            pass


# In[ ]:


random.shuffle(training_data)

X_test, y_test = [], []
for feature, label in training_data:
    X_test.append(feature)
    y_test.append(label)

X_test = np.array(X_test).reshape(-1, img_size, img_size, 1)
X_test = X_test / 255.0

y_test = to_categorical(y_test, num_classes = 10)


# # Model

# In[ ]:


model = Sequential()

model.add(Dropout(0.2, input_shape=(50,50,1)))

model.add(Conv2D(16, (5, 5)))
model.add(Activation("relu"))
model.add(BatchNormalization())

model.add(Conv2D(32, (5, 5)))
model.add(Activation("relu"))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25)) 


model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25)) 

model.add(Flatten())

model.add(Dense(100))
model.add(Activation("relu"))
model.add(BatchNormalization())

model.add(Dropout(0.5))  


model.add(Dense(10))
model.add(Activation("softmax"))


# # Complie and Callbacks

# In[ ]:


batch_size = 86
model_path = './Model.h5'


model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    EarlyStopping(monitor='val_acc', patience=20, mode='max', verbose=1),
    ModelCheckpoint(model_path, monitor='val_acc', save_best_only=True, mode='max', verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)
]


# # Image Generator
# * validation_split=0.2 in ImageDataGenerator, with subset='training'/subset='validation' in flow_from_directory
# ###### split the data to Train and Validation
# * color_mode="grayscale"
# ##### to make the input array (50,50,1)
# * try_except_gen
# ##### used when an error happen with data flow from directory, maybe there is a better way

# In[ ]:


train_datagen = ImageDataGenerator(rescale=1./255,
                                  validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
        '../input/notMNIST_large/notMNIST_large',
        target_size=(50, 50),
        batch_size= 265,
        color_mode="grayscale",
        class_mode='categorical',
        subset='training')

validation_generator = train_datagen.flow_from_directory(
        '../input/notMNIST_large/notMNIST_large',
        target_size=(50, 50),
        batch_size=265,
        color_mode="grayscale",
        class_mode='categorical',
        subset='validation')

def try_except_gen(gen):
    while True:
        try:
            data, labels = next(gen)
            yield data, labels
        except:
            pass


# # Model Training

# In[ ]:


history = model.fit_generator(try_except_gen(train_generator),
    epochs=30,
    validation_data=try_except_gen(validation_generator),
    steps_per_epoch= 529119 // 265,
    validation_steps= 18726 // 265,
    verbose= 1,
    callbacks = callbacks)


# # Visualize the training and testing accuracy

# In[ ]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, label='Training acc')
plt.plot(epochs, val_acc, label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss,  label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# # Model Evaluation on The test data

# In[ ]:


score = model.evaluate([X_test], [y_test], verbose=1)
print("Score: ",score[1]*100)

