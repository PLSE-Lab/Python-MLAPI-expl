#!/usr/bin/env python
# coding: utf-8

# # Hello and welcome
# 
# In this notebook you will get an example on how to create a CNN model using keras and tensorflow in backend. Dataset used is fashion mnist.

# Importing numpy and pandas.

# In[ ]:


import numpy as np
import pandas as pd


# # Loading the datasets.

# In[ ]:


train=pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')
test=pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')


# Splitting dataset into x_train, y_train, x_test and y_test

# In[ ]:


y_train=train['label']
y_test=test['label']
del train['label']
del test['label']


# In[ ]:


x_train=train.values
x_test=test.values


# In[ ]:


x_train.shape


# In[ ]:


x_test.shape


# # Reshaping the images.
# 28x28 pixel images from 784 columns

# In[ ]:


x_train=x_train.reshape(-1,28,28,1)
x_test=x_test.reshape(-1,28,28,1)


# In[ ]:


import matplotlib.pyplot as plt
plt.imshow(x_train[0][:,:,0])
plt.title(y_train[0])
plt.show()


# In[ ]:


import seaborn as sns


# Checking whether datast is balanced or not, we have 60000 images in train set from 10 classes and each class has 6000 images.

# In[ ]:


g = sns.countplot(y_train)

y_train.value_counts()


# In[ ]:


y_train.shape


# # Converting y(labels) into one hot vectors
# Using tensorflow.keras.utils.to_categorical() to convert y_train and y_test to onehot vectors.

# In[ ]:


from tensorflow.keras.utils import to_categorical
y_train=to_categorical(y_train, num_classes=10)
y_test=to_categorical(y_test, num_classes=10)
print(y_train[0])


# Normalisation of the inputs. Dividing by 255 makes values of all pixels between 0 and 1.

# In[ ]:


x_train=x_train/255
x_test=x_test/255


# In[ ]:


import tensorflow as tf
import keras
from keras import backend as K


# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU


# # Creating model.

# In[ ]:


model=Sequential()
model.add(Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)))
#model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(128,(3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


# In[ ]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# Ploting the model for better understanding.

# In[ ]:


import keras.utils
tf.keras.utils.plot_model(model, show_shapes=True)


# # Training the model

# In[ ]:


epochs=50
batch_size=600
history = model.fit(x_train, y_train,
                              epochs = epochs, verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size, validation_split=0.2)


# In[ ]:


plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()


# # Results
# Model gives an accuracy of 93.93% when used on the given test set.

# In[ ]:


score = model.evaluate(x_test, y_test, verbose=0)
print("Loss on test data",score[0])
print("Accuracy on test data", score[1]*100)

