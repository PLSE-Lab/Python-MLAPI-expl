#!/usr/bin/env python
# coding: utf-8

# In[54]:


# Imports
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


# In[5]:


os.listdir('../input/')


# In[8]:


# Load dataset
dataset = pd.read_json('../input/iris.json')
print('dataset shape:', dataset.shape)
dataset.head()


# In[10]:


# Seperate setosa, versicolor and virginica rows
setosa_ds = dataset[dataset['species'] == 'setosa']
virginica_ds = dataset[dataset['species'] == 'virginica']
versicolor_ds = dataset[dataset['species'] == 'versicolor']

print(setosa_ds.head())
print(virginica_ds.head())
print(versicolor_ds.head())


# In[19]:


# split total_train_rows rows from all the three classes and combine them to form a train dataset
# combine rest of the rows to form a test dataset
total_train_rows = 40

train_ds = pd.concat([setosa_ds[:total_train_rows], virginica_ds[:total_train_rows], versicolor_ds[:total_train_rows]], ignore_index=True)
test_ds = pd.concat([setosa_ds[total_train_rows:], virginica_ds[total_train_rows:], versicolor_ds[total_train_rows:]], ignore_index=True)


# In[22]:


print('train_ds shape:', train_ds.shape)
train_ds.head()


# In[23]:


print('test_ds shape:', test_ds.shape)
test_ds.head()


# In[28]:


# seperate features and labels from the datasets
train_X = train_ds.drop(columns=['species'])
train_Y = train_ds['species']

test_X = test_ds.drop(columns=['species'])
test_Y = test_ds['species']
train_X.head()


# In[31]:


train_Y.head()


# In[41]:


# create, train and evaluate the model
clf = KNeighborsClassifier()
clf.fit(train_X, train_Y)
clf.score(test_X, test_Y)


# In[89]:


# test prediction
index = np.random.randint(low=0, high=test_X.shape[0])
prediction = clf.predict(test_X[index:index+1])
print('prediction =', prediction)
print('actual =', test_Y[index])


# In[ ]:





# In[ ]:




