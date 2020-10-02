#!/usr/bin/env python
# coding: utf-8

# # Importing the Libraries

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf


# In[ ]:


# Checking the version of TensorFlow
tf.__version__


# # Loading the Dataset

# In[ ]:


train = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')
train.head()


# In[ ]:


# Creating a list of variables
features = train.columns.tolist()
features.remove('label')


# In[ ]:


test = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')
test.head()


# # Data Preprocessing

# ## Converting the Dataframes into NumPy arrays

# In[ ]:


X_train = np.array(train[features])
y_train = np.array(train['label'])


# In[ ]:


X_test = np.array(test[features])
y_test = np.array(test['label'])


# ## Normalizing the images
# 
# We divide the value of each pixel in the training and test datasets by the maximum value of pixels (255).
# 
# In this way, each pixel will be in the range of 0 and 1, which allows the ANN to train faster.

# In[ ]:


X_train = X_train / 255.0
X_test = X_test / 255.0


# # Building an Artificial Neural Network

# ## Defining the model
# 
# We define an object of the Sequential model.

# In[ ]:


model = tf.keras.models.Sequential()


# ## Adding a first Dense or Fully-connected hidden layer
# 
# Layer hyper-parameters:
# * Number of neurons = 128
# * Activation Function = ReLU
# * Input Shape = (784,)

# In[ ]:


model.add(tf.keras.layers.Dense(units = 128,activation = 'relu',input_shape = (784,)))


# ## Adding a second layer with Dropout
# 
# Dropout is a Regularization technique where we randomly set neurons in a layer to zero which means that during training those neurons will not be updated. This reduces the chance for overfitting.

# In[ ]:


model.add(tf.keras.layers.Dropout(0.2))


# ## Adding the output layer
# 
# * Number of neurons = 10 (corresponding to the number of classes)
# * Activation Function = Softmax

# In[ ]:


model.add(tf.keras.layers.Dense(units = 10,activation = 'softmax'))


# ## Compiling the model
# 
# * Optimizer = Adam
# * Loss = Sparse Categorical Crossentropy

# In[ ]:


model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['sparse_categorical_accuracy'])


# In[ ]:


model.summary()


# ## Training the model

# In[ ]:


model.fit(X_train,y_train, epochs = 10)


# ## Model evaluation and prediction

# In[ ]:


test_loss, test_acc = model.evaluate(X_test, y_test)


# In[ ]:


print('Test Accuracy : {}%'.format(test_acc*100))

