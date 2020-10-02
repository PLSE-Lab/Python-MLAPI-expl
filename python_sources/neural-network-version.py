#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers import Conv2D,Dropout,Dense,Flatten
from keras.models import Sequential

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("../input/train.csv", header=None)


# In[ ]:


test = pd.read_csv("../input/test.csv", header=None)


# In[ ]:


train_labels = pd.read_csv("../input/trainLabels.csv", header=None)


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


y = train_labels


# In[ ]:


X = train


# In[ ]:


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initializing Neural Network
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 128, init = 'uniform', activation = 'relu', input_dim = 40))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 64, init = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling Neural Network
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[ ]:


# Fitting our model 
classifier.fit(X, y, batch_size = 10, nb_epoch = 500)


# In[ ]:


from sklearn.metrics import mean_absolute_error
# Predicting the Test set results
y_pred = classifier.predict(X)
val_mae = mean_absolute_error(y_pred, y)


# In[ ]:


val_mae


# In[ ]:


test_y = classifier.predict(test)


# In[ ]:


test_y = np.array(test_y, dtype=int)


# In[ ]:


index_column = np.arange(1,9001)


# In[ ]:


myDataFrame = pd.DataFrame(test_y,columns=['Solution'],dtype=int)


# In[ ]:


myDataFrame['Id'] = index_column


# In[ ]:


myDataFrame.index.name = 'Id'


# In[ ]:


myDataFrame.set_index(index_column)


# In[ ]:


myDataFrame.to_csv("submission.csv", index=False)

