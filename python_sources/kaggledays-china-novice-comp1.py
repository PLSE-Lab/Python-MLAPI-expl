#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import math
import tensorflow as tf
print(tf.__version__)

np.random.seed(2019)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load and explore data

# In[ ]:


data_train_file = "../input/digit-recognizer/train.csv"
data_test_file = "../input/digit-recognizer/test.csv"

df_train = pd.read_csv(data_train_file)
df_test = pd.read_csv(data_test_file)


# In[ ]:


df_train.head()


# In[ ]:


# Notice data is shifted to the left by one column, since the label is missing
df_test.head()


# In[ ]:


df_test.describe()


# ## Preprocess data into numpy

# I've created a helper function to take care of processing the features into numpy arrays, but there are many options here.

# In[ ]:


# Note this returns numpy arrays
def get_features_labels(df):
    # The first column is the label.
    labels = df['label'].values
    
    # Select all columns except the first
    features = df.values[:, 1:]/255
    
    return features, labels


# In[ ]:


train_features, train_labels = get_features_labels(df_train)
test_features = df_test.values/255


# Confirm that the shape is what we expect: 42k in train, 28k in test, with 784 pixels per row

# In[ ]:


print(train_features.shape)
print(test_features.shape)
print(train_labels.shape)


# ## Visualize the numbers
# A helper function for visualizing a specific row

# In[ ]:


# Defaults to showing data from the training set, 
# but we can provide the test data as well, and leave labels as None, to visualize test set
def display_by_index(index, features=train_features, labels=train_labels):
    plt.figure()
    
    if labels is not None:
        plt.title(f'Label: {labels[index]}')
        
    _ = plt.imshow(np.reshape(features[index, :], (28,28)), 'gray')


# In[ ]:


# Visualize a training sample
display_by_index(221)


# In[ ]:


# Visualize a test sample
display_by_index(221, features=test_features, labels=None)


# Dataset looks fairly balanced. No need to do additional work here.

# In[ ]:


df_train['label'].value_counts()


# In[ ]:


print(df_test.shape)


# Turning labels into 1-hot encoding transforms the shape from 1 column to 10 columns

# In[ ]:


train_labels_1hot = tf.keras.utils.to_categorical(train_labels)


# In[ ]:


print(train_labels_1hot.shape)


# In[ ]:


train_labels_1hot


# ## Creating the model 
# For this one we use a deep neural net, expecting "okay" results, but nothing spectacular.
# 

# In[ ]:


model_arch = {}


# In[ ]:


model_arch['single_layer'] = [
      tf.keras.layers.Input(shape=(28*28,)),
      tf.keras.layers.Dense(10, activation='softmax')
  ]


# In[ ]:


model_arch['dnn'] = [
      tf.keras.layers.Input(shape=(28*28,)),
      tf.keras.layers.Dense(200, activation='sigmoid'),
      tf.keras.layers.Dense(60, activation='sigmoid'),
      tf.keras.layers.Dense(10, activation='softmax')
  ]


# In[ ]:


model_arch['dnn_relu'] = [
      tf.keras.layers.Input(shape=(28*28,)),
      tf.keras.layers.Dense(200, activation='relu'),
      tf.keras.layers.Dense(60, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
  ]


# ### Learning rate decay

# In[ ]:


# lr decay function
def lr_decay(epoch):
    return 0.01 * math.pow(0.6, epoch)

# lr schedule callback
lr_decay_callback = tf.keras.callbacks.LearningRateScheduler(lr_decay, verbose=True)

# Plot the decay rate
x = []
y = []
for i in range(1,10):
    y.append(lr_decay(i))
    x.append(i)
plt.plot(x, y)


# In[ ]:


# Add dropout
model_arch['dnn_relu_dropout'] = [
      tf.keras.layers.Input(shape=(28*28,)),
      tf.keras.layers.Dense(200, activation='relu'),
      tf.keras.layers.Dropout(0.25),
      tf.keras.layers.Dense(60, activation='relu'),
      tf.keras.layers.Dropout(0.25),
      tf.keras.layers.Dense(10, activation='softmax')
  ]


# In[ ]:


# CNN
model_arch['cnn'] = [
      tf.keras.layers.Reshape(input_shape=(28*28,), target_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(kernel_size=3, filters=12, activation='relu', padding='same'),
      tf.keras.layers.Conv2D(kernel_size=6, filters=24, activation='relu', padding='same', strides=2),
      tf.keras.layers.Conv2D(kernel_size=6, filters=32, activation='relu', padding='same', strides=2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(200, activation='relu'),
      tf.keras.layers.Dropout(0.25),
      tf.keras.layers.Dense(10, activation='softmax')
  ]


# ### Choose model architecture and compile

# In[ ]:


model_arch.keys()


# In[ ]:


model = tf.keras.Sequential(model_arch['cnn'])

# optimizer = 'sgd'
optimizer = 'adam'
# optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

# We will now compile and print out a summary of our model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.summary()


# ## Training the model
# Adjust the hyper params as needed.

# In[ ]:


BATCH_SIZE=128
EPOCHS=5


# In[ ]:


history = model.fit(train_features, train_labels_1hot, 
          epochs=EPOCHS, 
          batch_size=BATCH_SIZE, 
          validation_split=0.2)
#           ,callbacks=[lr_decay_callback])


# In[ ]:


plt.plot(history.history['acc'], color='b', label="Training accuracy")
plt.plot(history.history['val_acc'], color='r', label="Validation accuracy")
plt.legend(loc='lower right', shadow=True)


# ## Submission

# In[ ]:


predictions = model.predict_classes(test_features)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("submission.csv", index=False, header=True)


# ## Spot-checking some values

# In[ ]:


for i in range(200,210):
    display_by_index(i, features=test_features, labels=submissions["Label"].values)


# In[ ]:




