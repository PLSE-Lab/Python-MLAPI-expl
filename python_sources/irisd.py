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


# In[ ]:


#importing relevant libraries
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from sklearn.model_selection import train_test_split


# In[ ]:


#converting to label to one hot vector
def convert_to_oh(Ys):
    Y = np.zeros((len(Ys),3))
    for i in range(len(Ys)):
        if Ys[i]=='Iris-setosa':
            Y[i][0]=1
        elif Ys[i]=='Iris-versicolor':
            Y[i][1]=1
        else:
            Y[i][2]=1
    return Y


# In[ ]:


#defining the model
def iris_model(input_shape):
    X_input = Input(input_shape)
    X = layers.Dense(12, activation='relu', name='fc1')(X_input)
    #X = layers.Dropout(0.3)(X)
    X = layers.Dense(12, activation='relu', name='fc2')(X)
    X = layers.Dense(3, activation='softmax', name='pred')(X)
    
    model = Model(inputs=X_input, outputs=X, name='iris_model')
    return model


# In[ ]:


#loading and preprocessing the data
file = pd.read_csv('../input/iris/Iris.csv')
#file.describe()
#file.head()
Ys = np.array(file['Species'])
X = np.array(file.drop(['Species','Id'],1))
Y = convert_to_oh(Ys)
train_X, dev_X, train_Y, dev_Y = train_test_split(X, Y, test_size=0.1, random_state=1)
print(X.shape)
print(Y.shape)
print(train_X.shape)
print(train_Y.shape)


# In[ ]:


#preparing the model
model = iris_model(input_shape=(4))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# In[ ]:


#fitting and testing the model
model.fit(train_X, train_Y, epochs=50, batch_size=15, validation_data=(dev_X,dev_Y))


# In[ ]:


#saving the model
model.save('iris_mod')


# In[ ]:




