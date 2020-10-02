#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.


# In[2]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[4]:


from keras.utils import to_categorical
data_train = pd.read_csv('../input/fashion-mnist_train.csv')
data_test = pd.read_csv('../input/fashion-mnist_test.csv')


# In[ ]:


X_train = data_train.iloc[:,1:785].values
X_test = data_test.iloc[:,1:785].values


# In[18]:


y_train = data_train.iloc[:,0].values
y_test = data_test.iloc[:,0].values


# In[21]:


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# In[22]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[31]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


classifier = Sequential()
classifier.add(Dense(output_dim = 392, init = 'uniform', activation = 'relu', input_dim = 784))
classifier.add(Dense(output_dim = 196, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 98, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 1)


# In[ ]:


y_pred = classifier.predict_classes(X_test)
y_pred

