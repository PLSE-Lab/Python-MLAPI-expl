#!/usr/bin/env python
# coding: utf-8

# Hello Everyone
# This is my first notebook on Deep Learning using Keras API on MNIST Datasets.
# I am learning about deep Learning and Keras API and doing application work on Kaggle to know to deal with problems.
# Thanks to the Learning from Datacamp.
# 

# ## Deep Learning Using Keras ##

# Firstly we will import some libraries required to solve the problem. Datasets are in ../input/ directory.
# We will import Keras Libarary needed to set neural net for MNIST hand written datasets. We will be using Adam optimizer and develop  a Sequential model.
# 

# In[ ]:


import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = (pd.read_csv("../input/test.csv").values).astype('float32')


# In[ ]:


train.head()


# Train Datasets contain the first column as the label which describes the class to which data of entire row belongs to. Therefore we will slice the datasets to two part 
# 1. labels- contain the first column
# 2. train_img- contain the rest data of entire column

# In[ ]:


train_img = (train.ix[:,1:].values).astype('float32')
labels = train.ix[:,0].values.astype('int32')


# In[ ]:


train_img.shape


# In[ ]:


test.shape


# In[ ]:


labels


# Now we will convert the labels into categorical type of data which implies we needed to have output nodes equal to the numbers of categorical label. Here in digits datasets we require only 10 output nodes

# In[ ]:


labels = to_categorical(labels)
classes = labels.shape[1]
classes


# In[ ]:


plt.plot(labels[9])
plt.xticks(range(10));


# In[ ]:


# fix random seed for reproducibility
seed = 43
np.random.seed(seed)


# In[ ]:


labels.shape


# 
# 
# Neural Network
# --------------

# Now we will design the **Neural Network Architecture.**
# Here we will develop the Sequential Model,  and model.add() will help me to add the layer of the neuron of different units in the network. We are using activation function **Rectifier** in the input layer and Hidden layer. **Softmax function** in the output layer that gives the output in 10 different nodes of output layer.

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense , Dropout

model=Sequential()
model.add(Dense(32,activation='relu',input_dim=(28 *28)))
model.add(Dense(16,activation='relu'))
model.add(Dense(10,activation='softmax'))


# In[ ]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


model_fit = model.fit(train_img, labels, validation_split = 0.05, epochs=24, batch_size=64)


# In[ ]:


pred = model.predict_classes(test)

result = pd.DataFrame({"ImageId": list(range(1,len(pred)+1)),"Label": pred})
result.to_csv("output.csv", index=False, header=True)


# In[ ]:




