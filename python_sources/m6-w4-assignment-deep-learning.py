#!/usr/bin/env python
# coding: utf-8

# # DEEP LEARNING

#     'The MNIST dataset is a very famous dataset used to test and benchmark new deep learning architectures and models. It contains images of handwritten digits (from 0 to 9). It consists of a training and test sets of features and labels.'

# ## GOAL

# 1) Use Keras and develop a model that correctly detects a handwritten digit
# 
# 2) Evaluate the model properly and interpret its performance

# ## OUTLINE

# * PRE PROCESSING AND DATA CLEANING
# * MODEL BUILDING
# * MODEL ASSESSMENT

# ## PRE PROCESSING AND DATA CLEANING

# We will start by importing all required modules:

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#model building modules
from keras.models import Sequential # initial instance of model
from keras.layers import Dense # layers 
from keras.utils import np_utils #OneHotEncoding


# ... and load our train and test data.

# In[ ]:


train = pd.read_csv('../input/mnist-in-csv/mnist_train.csv') # load train
test = pd.read_csv('../input/mnist-in-csv/mnist_test.csv') # load test


# Lets check the content:

# In[ ]:


train.head(1) # train head


# In[ ]:


test.head(1) # test head


#     'Each row consists of 785 values: the first value is the label (a number from 0 to 9) and the remaining 784 values are the pixel values (a number from 0 to 255)'

# In[ ]:


print(train.info())
print(test.info())


# All variables are integer type.

# ### Data Preparation

# As we are dealing with multi-class target, and our target is already numerical, we need to dummify it. The target here is the label (0-9).

# In[ ]:


def lab(df):
    lab_dum = np_utils.to_categorical(df['label']) # convert label to dummy variable
    return lab_dum


# In[ ]:


y_train = lab(train)
y_test = lab(test)


# In[ ]:


X_train = train.iloc[:,1:] # create X_train
X_test = test.iloc[:,1:]# create X_test

# normalize X Train and X test as they are between 0 and 255
X_train /= 255
X_test /= 255


# In[ ]:


X_train.info()


# ## MODEL BUILDING

# In[ ]:


# create a function to instanciate, add layers, compile the model
def nnmod():
	# instanciate and add layers to model
	nnmod = Sequential()
	nnmod.add(Dense(784, input_dim=784, activation='relu'))
	nnmod.add(Dense(10, activation='softmax'))
	# compile model
	nnmod.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return nnmod


# In[ ]:


nnmod = baseline_model() # run our function

nnmod.fit(X_train, y_train, batch_size=200, epochs=10, validation_data=(X_test, y_test)) # fit model


# ## MODEL ASSESSMENT

# We are using accuracy to evaluate how the model is performing. Therefore:

# In[ ]:


accuracy = nnmod.evaluate(X_test, y_test)
print("Accuracy is: ", score[1]*100, "%")


# This score is more or less good, but we an improve by adding some other layers.
