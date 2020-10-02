#!/usr/bin/env python
# coding: utf-8

# # Using the Wisconsin breast cancer diagnostic data set for predictive analysis
# ## Priya Theru
# 
# ANN

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


dataset = pd.read_csv("../input/data.csv",header = 0)
dataset.head()


# In[ ]:



X= dataset.iloc[:,2:32].values
Y = dataset.iloc[:,1].values


# In[ ]:


X


# In[ ]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)


# In[ ]:


print(X.shape)


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# In[ ]:


import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# In[ ]:


classifier = Sequential()
classifier.add(Dense(units = 16,input_dim = 30, kernel_initializer = 'uniform', activation = 'relu'))
##classifier.add(Dropout(p=0.1))
classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
##classifier.add(Dropout(p=0.1))
classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
##classifier.add(Dropout(p=0.1))
classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
##classifier.add(Dropout(p=0.1))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


# In[ ]:


classifier.compile(optimizer = 'adam' ,loss  = 'binary_crossentropy' ,metrics = ['accuracy'])


# In[ ]:


classifier.fit(x_train, y_train, batch_size = 30, epochs = 100)


# In[ ]:


y_pred = classifier.predict(x_test)
y_pred = y_pred > 0.5
from sklearn.metrics import confusion_matrix
confusion_matrix (y_pred, y_test)


# Conclusion -
# 
# 96% accuracy
