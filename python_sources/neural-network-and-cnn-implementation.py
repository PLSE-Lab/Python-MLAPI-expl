#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd

from keras.layers import Dense, Dropout, Activation, Reshape, Conv2D, AveragePooling2D, Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
from keras.optimizers import adam

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


ds = pd.read_csv('../input/train.csv')
data = ds.values


# **Split and Normalize the data**
# 
# Split 'train.csv' into 2 sets: Training Set consists of 80% rows Validation Set consists of 20%
# 
# Normalize the data by dividing it by maximum value of a pixel i.e. 255.0 (255.0 such that the answer is stored as a float)

# In[ ]:


split = int(0.8 * data.shape[0])

X_train = data[:split, 1:]/255.0
y_train = np_utils.to_categorical(data[:split, 0])

X_val = data[split:, 1:]/255.0
y_val = np_utils.to_categorical(data[split:, 0])

print (X_train.shape, y_train.shape)
print (X_val.shape, y_val.shape)


# 
# **Feed Forward Model Using Keras**

# model = Sequential()
# 
# model.add(Dense(512, input_shape=(784, )))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# 
# model.add(Dense(180))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# 
# model.add(Dense(10))
# model.add(Activation('softmax'))
# model.summary()

# model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

# I ran it for 30 epochs because the value starts to plateau after the 30th epoch giving out the maximum accuracy. 
# 
# I would recommend to try and test it yourself the number of epochs you want to train, and the number of layers (and nodes in each layer of your model).
# This is intuitive and there is no specific rule for it.
# 
# And running the model on train and validation set and saw that the model is not getting overfitted since the validation_accuracy is very similar to Accuracy. So now I will train the model on the whole 'train.csv' dataset.
# 

# X_train = data[:, 1:]/255.0
# y_train = np_utils.to_categorical(data[:, 0])
# hist = model.fit(X_train, y_train,
#                  epochs = 30,
#                  shuffle = True,
#                  batch_size=128
#                  ,validation_data=(X_val, y_val)
#                  )

# Now that I know that there is not much of a difference b/w Accuracy and Validation_Accuracy, I can now fit the model to the whole data i.e. 'train.csv' and make predictions on the 'test.csv'

# test = pd.read_csv('../input/test.csv')
# sub = model.predict(test)
# result = np.argmax(sub,axis = 1)

# This model secured in the top 58%.
# 
# Now I will try implementing Convolution Neural Networks and see how my rank increases with the power of CNN's.
# 
# 
# > **ConvNet**

# In[ ]:


model = Sequential()
model.add(Reshape(target_shape=(1, 28, 28), input_shape = (784,)))

model.add(Conv2D(kernel_size = (5, 5), filters = 4, padding = "same"))

model.add(Conv2D(kernel_size = (5, 5), filters = 8, padding = "same"))
model.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_first"))

model.add(Conv2D(kernel_size = (5, 5), filters = 16, padding = "same"))

model.add(Conv2D(kernel_size = (5, 5), filters = 32, padding = "same"))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

model.add(Conv2D(kernel_size = (5, 5), filters = 128, padding = "same"))

model.add(Flatten())

model.add(Dense(output_dim = 128, activation = 'relu'))

model.add(Dense(output_dim = 100, activation = 'relu'))
# Need an output of probabilities of all the 10 digits
model.add(Dense(output_dim = 10, activation= 'softmax'))


# In[ ]:


# adam = keras.optimizers.Adam(lr = 0.0005, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

X_train = data[:, 1:]/255.0
y_train = np_utils.to_categorical(data[:, 0])

model.fit(X_train, y_train, epochs = 30, batch_size=64)


# In[ ]:


test = pd.read_csv('../input/test.csv')
result = model.predict(test)


# In[ ]:


result = np.argmax(result,axis = 1)
result = pd.Series(result,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),result],axis = 1)
submission.to_csv("submission_ANN.csv",index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




