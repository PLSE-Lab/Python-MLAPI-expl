#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# This notebook contains a simple CNN architecture which will be trained to perform digit recognition. The following parts will be considered:
# * Converting CSV file data to an image representation (square matrix) using Numpy 
# * Visualising the dataset using MatPlotLib
# * Defining a CNN architecture using Keras
# * Using early stopping as part of a Keras callback to stop training early when accuracy stops increasing
# * Plotting curves of accuracy and validation accuracy
# * Predicting the classifications of a test dataset
# * Compile the results into a specified .csv format
# 
# These steps can be applied to a variety of data science problems.

# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Input, Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from matplotlib import pyplot


# The training data is in the form of a csv file where each digit image is represented by 784 columns. Hence, need to reformat this data (to 28x28) to allow a CNN to operate on it as it would with an image.

# In[ ]:


train_dir = '../input/digit-recognizer/train.csv'

data = pd.read_csv(train_dir)
# extract x,y training data
y_train = data['label']
y_train = to_categorical(y_train.values, num_classes=10)
print(y_train.shape)
x_train = data.drop(labels = ['label'], axis = 1)
x_train = x_train.to_numpy()
# normalise values into [0,1] range to speed up training
x_train = x_train/255
# resize to represent 28x28 image
# x_train shape changes from (42000,784) to (42000, 28, 28, 1) 
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
print(x_train.shape)


# Let's now visualise some of the images in the training set. Here we display the first 9 images in a 3x3 grid. You can see the variation between 1's and 0's which will need to be learnt by the classifier. 

# In[ ]:


for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(x_train[i][:,:,0])


# In[ ]:


test_dir = '../input/digit-recognizer/test.csv'
test = pd.read_csv(test_dir)
test = test.to_numpy()
test = test/255
test = test.reshape(test.shape[0], 28, 28, 1)


# Define the CNN architecture. Due to the small input images, I have chosen a relatively shallow CNN with small kernel sizes (2,2). Categorical cross entropy is chosen as the loss function as it works well for multi-class classification problems. Dropout is used to prevent overfitting.

# In[ ]:


simple_cnn = Sequential()
simple_cnn.add(Conv2D(filters=20, kernel_size=(2, 2), activation='relu', input_shape=(28,28,1)))
simple_cnn.add(MaxPool2D())
simple_cnn.add(Conv2D(filters=20, kernel_size=(2, 2), activation='relu'))
simple_cnn.add(MaxPool2D())
simple_cnn.add(Flatten())
simple_cnn.add(Dense(units = 120, activation = 'relu'))
simple_cnn.add(Dense(units = 10, activation = 'softmax'))

simple_cnn.compile('adam', 'categorical_crossentropy',metrics = ['acc'])
simple_cnn.summary()


# Early stopping means that the model stops training if the validation accuracy hasn't increased in n training epochs (here n = 10). It's included as a callback in model.fit() so it is called on every training cycle.

# In[ ]:


# Early stopping dependent on validation accuracy
es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=10)

history = simple_cnn.fit(x_train, y_train, validation_split = 0.2, epochs = 500, callbacks = [es])


# Plot the results so underfitting/overfitting problems can be easily diagnosed.

# In[ ]:


pyplot.plot(history.history['acc'], label='train')
pyplot.plot(history.history['val_acc'], label='val')
pyplot.legend()
pyplot.show()


# Predict the values on the test dataset using the newly trained model.

# In[ ]:


preds = simple_cnn.predict(test)
preds = np.argmax(preds,axis = 1)
preds


# Compile the results in the specified format (as given in sample_submission.csv)

# In[ ]:


results = pd.DataFrame()
results['ImageId'] = np.arange(len(preds)) + 1
results['Label'] = pd.Series(preds)
results
# index false so we don't write row names
results.to_csv('submission.csv', index = False)

