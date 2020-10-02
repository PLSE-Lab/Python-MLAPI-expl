#!/usr/bin/env python
# coding: utf-8

# # An Awesome Monkeys Classificator (Deep Learning)
# - transfer learning + inception v3

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("../input/10-monkey-species"))

# Any results you write to the current directory are saved as output.


# ## Data Processing
# ### Prepare Data
# load the data

# In[ ]:


train_dir = '../input/10-monkey-species/training/training/'
val_dir = '../input/10-monkey-species/validation/validation/'
labels = pd.read_csv("../input/10-monkey-species/monkey_labels.txt")
num_classes = labels['Label'].size
labels


# ### Data Augmentation
# 
# - flip
# - width_shift
# - height_shift
# 

# In[ ]:




#Fit the model using Data Augemnetation, will improve accuracy of the model
from tensorflow.python.keras.applications.inception_v3 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                                              horizontal_flip = True,
                                              
                                              rotation_range=20,
                                              width_shift_range = 0.2,
                                              height_shift_range = 0.2)
            
data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input)


# ## Set up Model
# 

# In[ ]:


from tensorflow.python.keras.applications import InceptionV3
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

num_clasees =10

model = Sequential()
model.add(InceptionV3(include_top=False, 
                      pooling='avg', 
                      weights='../input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
                     ))
model.add(Dense(num_clasees, activation="softmax"))

# do not need to train the pre train layer
model.layers[0].trainable=False


# In[ ]:


model.layers


# In[ ]:


model.summary()


# ### Model Compile

# In[ ]:



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# ### Model Fit
# 

# In[ ]:


image_size =512

train_generator = data_generator_with_aug.flow_from_directory(
       directory = train_dir,
       target_size=(image_size, image_size),
       batch_size=24,
       class_mode='categorical')

validation_generator = data_generator_no_aug.flow_from_directory(
       directory = val_dir,
       target_size=(image_size, image_size),
       class_mode='categorical')


# In[ ]:


model_history = model.fit_generator(
        train_generator,
        steps_per_epoch=46,
        epochs=32,
        validation_data=validation_generator,
        validation_steps=1)


# ## Save Model

# In[ ]:


model.save('incept_ez.h5')  # creates a HDF5 file 'my_model.h5'


# In[ ]:


import pandas as pd
history = pd.DataFrame()
history["acc"] = model_history.history["acc"]
history["val_acc"] = model_history.history["val_acc"]
history.plot(figsize=(12, 6))


# In[ ]:


acc = model_history.history['acc']
val_acc = model_history.history['val_acc']
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
epochs = range(1, len(acc) + 1)

import matplotlib.pyplot as plt

plt.title('Training and validation accuracy')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, val_acc, 'blue', label='Validation acc')
plt.legend()

plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')

plt.legend()

plt.show()


# ## Thanks and Credits
# 
# - Dan Becker https://www.kaggle.com/learn
# - Juan https://www.kaggle.com/moriano/monkey-species-transfer-learning-95-6-accuracy
# - Dan Rusei https://www.kaggle.com/danrusei/10-monkey-keras-transfer-learning-resnet50
# 
