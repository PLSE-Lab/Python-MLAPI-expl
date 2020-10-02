#!/usr/bin/env python
# coding: utf-8

# Hand-written digit recognizer using LeNet-5 Convolutional Neural Network achitecture.
# 
# The training set is the MNIST dataset.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Keras stuff
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import backend as k_backend


# In[ ]:


file_path = '/kaggle/input/digit-recognizer/'


# In[ ]:


# load traing and test sets
train = pd.read_csv(file_path + 'train.csv')
test = pd.read_csv(file_path + 'test.csv')


# In[ ]:


# separate the labels column from the training set
Y_train = train[['label']]
X_train = train.drop(train.columns[[0]], axis=1)
X_test = test


# In[ ]:


# have a glance on the data
sample = X_train.iloc[0, :]
sample = sample.values.reshape([28, 28])
plt.imshow(sample, cmap='gray')
print('This is: ' + str(Y_train.iloc[1000, 0]))


# In[ ]:


X_train = np.array(X_train)
X_test = np.array(X_test)


# In[ ]:


# reshape the images in the train and test sets to thier original sizes
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)


# In[ ]:


# the original LeNet-5 describes a 32*32 input images. So let's pad ours
X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')


# In[ ]:


# standardize the training set to keep weights under control during optimization
mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)
X_train = (X_train - mean_px) / std_px


# In[ ]:


# One-hot-encode the training set labels
Y_train = to_categorical(Y_train)


# In[ ]:


# create the input layer
X_input = Input(shape=(32, 32, 1))


# In[ ]:


# the first block of layers: conv1 + maxpool1
conv1 = Conv2D(filters=6, kernel_size=5, strides=(1, 1), activation='relu')(X_input)
max_pool1 = MaxPooling2D(pool_size=2, strides=2)(conv1)


# In[ ]:


# the second block of layers: conv2 + maxpool2
conv2 = Conv2D(filters=16, kernel_size=5, strides=(1, 1), activation='relu')(max_pool1)
max_pool2 = MaxPooling2D(pool_size=2, strides=2)(conv2)


# In[ ]:


# flatten the input, then apply two standard fully connected layers
flat = Flatten()(max_pool2)
dense1 = Dense(units=120, activation='relu')(flat)
dense2 = Dense(units=84, activation='relu')(dense1)


# In[ ]:


# the output layers
out_enc = Dense(units=10, activation='softmax')(dense2)


# In[ ]:


# create and compile the model
digit_recognizer = Model(inputs=X_input, outputs=out_enc)
digit_recognizer.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


# train and save the model
digit_recognizer.fit(X_train, Y_train, epochs=42, steps_per_epoch=31)
digit_recognizer.save('digit_recognizer_model_lenet5.h5')


# In[ ]:


# use the trained model to predict the labels of the test set
y_pred = digit_recognizer.predict(X_test)


# In[ ]:


# decode the labels and concatenate with image IDs
labels = np.argmax(y_pred, axis=1)
labels = labels.reshape([len(labels), 1])
index = np.arange(1, len(y_pred) + 1)
index = index.reshape([len(index), 1])
output = np.concatenate([index, labels], axis=1)


# In[ ]:


# write predictions to the disk. Cheers!
np.savetxt('digit_recognizer_predictions.csv', output, delimiter=',', fmt="%s", header='ImageId,Label', comments='')

