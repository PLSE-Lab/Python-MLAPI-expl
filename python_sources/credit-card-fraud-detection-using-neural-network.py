#!/usr/bin/env python
# coding: utf-8

# ANN ( Artificial Neural Networks ) are great for many of the problems , they can make a clear relation between  variables , after all it's just matrix multiplication
# i.e Linear Algebra and they are easy to implement using Frameworks like Keras.
# Keras is a great framework when it comes to implementing Neural networks . 
# Its just a wrapper around tensorflow . 

# In[2]:


#importing libraries

import pandas as pd
import numpy as np


# In[3]:


# Reading and cleaning dataset

dataset = pd.read_csv('../input/creditcard.csv')
dataset.dropna()
dataset.head()


# In[4]:


#Getting the no of columns

dataset.columns


# In[5]:


# Getting the X_data

X_data = dataset.iloc[:,0:-1].values
X_data[0:2]


# In[6]:


#Getting the Y_data

Y_data = dataset.iloc[:,-1].values
Y_data[0:5]


# In[7]:


# Getting no of instances of each unique class

unique , counts = np.unique(Y_data,return_counts = True)
print(unique,counts)


# In[8]:


# here 0 means not spam and 1 represents spam


# In[9]:


# importing StandardScaler used for scaling

from sklearn.preprocessing import StandardScaler
sclaer = StandardScaler()


# In[11]:


# Scaling X_data

X_data = sclaer.fit_transform(X_data)
X_data[0:2]


# In[12]:


# Splitting the data into training and testing  

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X_data,Y_data,test_size=0.3)


# In[14]:


# Libraries that we will need to create ANN

from keras.models import Sequential
from keras.layers import Dense


# In[15]:


# building our Neural Network

classifier = Sequential()
classifier.add(Dense(40 , input_dim = 30 , activation = 'relu'))
classifier.add(Dense(30 , input_dim = 40 , activation = 'relu'))
classifier.add(Dense(20 , input_dim = 30 , activation = 'relu'))
classifier.add(Dense(10 , input_dim = 20 , activation = 'relu'))
classifier.add(Dense(6 , input_dim = 10 , activation = 'relu'))
classifier.add(Dense(4 , input_dim = 6 , activation = 'relu'))
classifier.add(Dense(1, input_dim = 4 , activation = 'sigmoid'))


# In[16]:


# Specifying our loss and optimizer

classifier.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )


# In[17]:


# Fitting our data to ANN

classifier.fit( x_train , y_train , epochs = 200 , batch_size = 500 )


# In[18]:


# Getting the results

classifier.evaluate(x_test,y_test)


# **We got a test score of 99.9% using Artificial Neural Network ** 
