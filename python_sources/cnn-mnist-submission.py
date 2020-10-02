#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import losses,optimizers,metrics
from keras.utils.np_utils import to_categorical

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ### Load the data

# In[2]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[3]:


print('Shape of training data is {}.'.format(train.shape))
print('Shape of test data is {}.'.format(test.shape))


# In[4]:


train.head()


# Will only use a subset of these because the kernel seems to run out of memory and crash if I use the entire dataset.

# In[5]:


# # select a random sample of dataset without replacement
# train = train.sample(n=10000,replace=False,random_state=101)
# test = test.sample(n=2000,replace=False,random_state=101)


# In[6]:


# Check that we have about equal numbers in each class
train['label'].value_counts()


# Separate labels from images in training data

# In[7]:


y_train = train['label']
X_train = train.drop('label',axis=1)
print('Shapes of training labels and data are {} and {}.'.format(y_train.shape,X_train.shape))


# ### Preprocess
# Check the values of the data to see if normalisation is required.

# In[8]:


X_train.max().max(),X_train.min().min()


# Normalise by maximum so all images have values from 0 to 1.

# In[9]:


X_train /= 255
test /= 255


# In[10]:


# check to make sure
print(X_train.max().max(),X_train.min().min())
print('\n')
print(test.max().max(),test.min().min())


# Let's one hot encode the labels so we can use them in training

# In[11]:


y_train = to_categorical(y_train, num_classes = 10)
y_train.shape


# Create training and validation split from the training data. Let's do 80/20.

# In[12]:


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=101)


# In[13]:


print('Shapes of training labels and data are {} and {}.'.format(y_train.shape,X_train.shape))
print('\n')
print('Shapes of validation labels and data are {} and {}.'.format(y_val.shape,X_val.shape))


# Plot an image so we know what we are working with

# In[14]:


test_image = X_train.values.reshape(-1,28,28,1)
test_image.shape


# In[15]:


indx = 8 # change this to see different numbers
plt.imshow(test_image[indx][:,:,0],cmap='Greys')
print('What is the label? {}'.format(np.where(y_train[indx]==1)[0]))


# ### Model definition and implementation
# 
# Will be using a Convolution Neural Network for this

# In[16]:


cnn_kmodel = models.Sequential()


# In[17]:


cnn_kmodel.add(layers.Conv2D(filters = 32, kernel_size = (6,6),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
cnn_kmodel.add(layers.MaxPool2D(pool_size=(2,2)))
cnn_kmodel.add(layers.Dropout(0.25))

cnn_kmodel.add(layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
cnn_kmodel.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
cnn_kmodel.add(layers.Dropout(0.25))

cnn_kmodel.add(layers.Flatten())
cnn_kmodel.add(layers.Dense(512, activation = "relu"))
cnn_kmodel.add(layers.Dropout(0.5))
cnn_kmodel.add(layers.Dense(10, activation = "softmax"))


# In[18]:


# optimizer
opt = tf.keras.optimizers.Adam()


# In[19]:


cnn_kmodel.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])


# In[20]:


epochs = 10
batch_size = 50


# In[21]:


cnn_kmodel.fit(X_train.values.reshape(-1,28,28,1), y_train, batch_size=batch_size, epochs=epochs,
               validation_data=(X_val.values.reshape(-1,28,28,1), y_val), shuffle=True)


# ### Check model performance with validation data
# 
# Check Accuracy, Precision and Recall with the validation data set

# In[22]:


plt.figure(figsize=[9,6])
plt.plot(cnn_kmodel.history.history['loss'], color='b', label="Training loss")
plt.plot(cnn_kmodel.history.history['val_loss'], color='r', label="validation loss")
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('Loss')


# Need to predict on validation set and convert the validation one hot encoded labels back to their orignal form

# In[23]:


predict_vals = cnn_kmodel.predict_classes(X_val.values.reshape(-1,28,28,1))


# In[24]:


yval_label = np.argmax(y_val,axis=1) 
print(yval_label)


# In[25]:


print('Metric for Validation set')
print('Classification report:')
print(classification_report(predict_vals,yval_label))
print('\n')
print('Accuracy score is {:6.3f}.'.format(accuracy_score(predict_vals,yval_label)))


# In[26]:


plt.figure(figsize=[9,6])
sns.heatmap(confusion_matrix(predict_vals,yval_label),cmap='gist_gray',cbar=False,
            annot=True,fmt='d',linewidths=0.5)
plt.title('Confusion Matrix for Validation Set')


# All the measures are averaging about 99 % which is pretty good.
# 
# 
# ### Test data submission

# In[27]:


sub_results = cnn_kmodel.predict(test.values.reshape(-1,28,28,1))

sub_results = np.argmax(sub_results, axis=1)

submission = pd.DataFrame({'ImageId':np.arange(1,sub_results.size +1),'Label':sub_results})
submission.to_csv("submission.csv",index=False)


# Ok that's my first submission :D
