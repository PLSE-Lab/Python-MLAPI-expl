#!/usr/bin/env python
# coding: utf-8

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


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error
from tqdm import tqdm_notebook


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error
from tqdm import tqdm_notebook

from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_blobs #used to genereate non-linearly seperable data


# In[ ]:


train_data=pd.read_csv('/kaggle/input/titanic/train.csv')
# train_data_copy=read_csv()
test_data=pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


train_data.head()


# In[ ]:


train_data.shape


# In[ ]:


train_data.isnull().sum()


# In[ ]:


train_data.drop(columns='Cabin',axis=1,inplace=True)


# In[ ]:


train_data.isnull().sum()


# In[ ]:


train_data.Age.fillna(value=train_data.Age.mean(),inplace=True)


# In[ ]:


train_data.isnull().sum()


# In[ ]:


train_data.Embarked.unique()


# In[ ]:


train_data.loc[train_data.Embarked=='S'].count()


# In[ ]:


#maximum embarked have value S so lets replace null with S
train_data.loc[train_data.Embarked=='S']


# In[ ]:


train_data.loc[train_data.Embarked.isna()].Embarked


# In[ ]:


train_data.loc[train_data.Embarked.isna()].Embarked


# In[ ]:


train_data.Embarked.fillna(value='S',inplace=True)


# In[ ]:


train_data.loc[train_data.Embarked.isna()].Embarked


# In[ ]:


train_data.isna().sum()


# In[ ]:


#now our data has no missing values
train_data.head()


# In[ ]:


filtered_train_data=train_data[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Embarked']]


# In[ ]:


filtered_train_data.set_index('PassengerId',inplace=True)


# In[ ]:


filtered_train_data.head()


# In[ ]:


Y_train=train_data['Survived']


# In[ ]:


Y_train.head()


# In[ ]:


filtered_train_data.Pclass.unique()


# In[ ]:


filtered_train_data.Sex.unique()


# In[ ]:


filtered_train_data.Sex.replace({'male':1,'female':0},inplace=True)


# In[ ]:


filtered_train_data.head()


# In[ ]:


filtered_train_data.Embarked.unique()


# In[ ]:


filtered_train_data.Embarked.replace({'S':0,'C':1,'Q':2},inplace=True)


# In[ ]:


filtered_train_data.head()


# In[ ]:


filtered_train_data.Parch.unique()


# In[ ]:


min=filtered_train_data.Age.min()
max=filtered_train_data.Age.max()
print(min," ",max)


# In[ ]:


filtered_train_data.Age=(filtered_train_data.Age-min)/(max-min)


# In[ ]:


filtered_train_data.head()


# In[ ]:


# model=DecisionTreeClassifier()
model=RandomForestClassifier(n_estimators=10,max_leaf_nodes=10,random_state=0)


# In[ ]:


model.fit(filtered_train_data,Y_train)


# In[ ]:


test_data.head()


# In[ ]:


test_data.isnull().sum()


# In[ ]:


filtered_test_data=test_data[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Embarked']]


# In[ ]:


filtered_test_data.head()


# In[ ]:


filtered_test_data.set_index('PassengerId',inplace=True)


# In[ ]:


filtered_test_data.Sex.replace({'male':1,'female':0},inplace=True)


# In[ ]:


filtered_test_data.Age.fillna(value=filtered_test_data.Age.mean(),inplace=True)


# In[ ]:


test_min=filtered_test_data.Age.min()
test_max=filtered_test_data.Age.max()
print(max," ",min)


# In[ ]:


filtered_test_data.Age=(filtered_test_data.Age-test_min)/(test_max-test_min)


# In[ ]:


filtered_test_data.head()


# In[ ]:


filtered_test_data.Embarked.replace({'S':0,'C':1,'Q':2},inplace=True)


# In[ ]:


filtered_test_data.head()


# In[ ]:


Y_pred=model.predict(filtered_test_data)


# In[ ]:


# predicted_y=pd.DataFrame({'PassengerId':filtered_test_data.index,'Survived':Y_pred})


# In[ ]:


# predicted_y.head()


# In[ ]:


# predicted_y.to_csv('submission.csv',index=False)


# # FeedForwordNetwork

# In[ ]:


class FFSNNetwork:
  
  def __init__(self, n_inputs, hidden_sizes=[2]):
    self.nx = n_inputs
    self.ny = 1
    self.nh = len(hidden_sizes)
    self.sizes = [self.nx] + hidden_sizes + [self.ny]
    
    self.W = {}
    self.B = {}
    for i in range(self.nh+1):
      self.W[i+1] = np.random.randn(self.sizes[i], self.sizes[i+1])
      self.B[i+1] = np.zeros((1, self.sizes[i+1]))
  
  def sigmoid(self, x):
    return 1.0/(1.0 + np.exp(-x))
  
  def forward_pass(self, x):
    self.A = {}
    self.H = {}
    self.H[0] = x.reshape(1, -1)
    for i in range(self.nh+1):
      self.A[i+1] = np.matmul(self.H[i], self.W[i+1]) + self.B[i+1]
      self.H[i+1] = self.sigmoid(self.A[i+1])
    return self.H[self.nh+1]
  
  def grad_sigmoid(self, x):
    return x*(1-x) 
    
  def grad(self, x, y):
    self.forward_pass(x)
    self.dW = {}
    self.dB = {}
    self.dH = {}
    self.dA = {}
    L = self.nh + 1
    self.dA[L] = (self.H[L] - y)
    for k in range(L, 0, -1):
      self.dW[k] = np.matmul(self.H[k-1].T, self.dA[k])
      self.dB[k] = self.dA[k]
      self.dH[k-1] = np.matmul(self.dA[k], self.W[k].T)
      self.dA[k-1] = np.multiply(self.dH[k-1], self.grad_sigmoid(self.H[k-1]))
    
  def fit(self, X, Y, epochs=1, learning_rate=1, initialise=True, display_loss=False):
    
    # initialise w, b
    if initialise:
      for i in range(self.nh+1):
        self.W[i+1] = np.random.randn(self.sizes[i], self.sizes[i+1])
        self.B[i+1] = np.zeros((1, self.sizes[i+1]))
      
    if display_loss:
      loss = {}
    
    for e in tqdm_notebook(range(epochs), total=epochs, unit="epoch"):
      dW = {}
      dB = {}
      for i in range(self.nh+1):
        dW[i+1] = np.zeros((self.sizes[i], self.sizes[i+1]))
        dB[i+1] = np.zeros((1, self.sizes[i+1]))
      for x, y in zip(X, Y):
        self.grad(x, y)
        for i in range(self.nh+1):
          dW[i+1] += self.dW[i+1]
          dB[i+1] += self.dB[i+1]
        
      m = X.shape[1]
      for i in range(self.nh+1):
        self.W[i+1] -= learning_rate * dW[i+1] / m
        self.B[i+1] -= learning_rate * dB[i+1] / m
      
      if display_loss:
        Y_pred = self.predict(X)
        loss[e] = mean_squared_error(Y_pred, Y)
    
    if display_loss:
      plt.plot(np.array(list(loss.values())).astype(float))
      plt.xlabel('Epochs')
      plt.ylabel('Mean Squared Error')
      plt.show()
      
  def predict(self, X):
    Y_pred = []
    for x in X:
      y_pred = self.forward_pass(x)
      Y_pred.append(y_pred)
    return np.array(Y_pred).squeeze()


# In[ ]:


converted_filtered_train_data=np.array(filtered_train_data)
converted_y_train=np.array(Y_train)
print(converted_filtered_data,converted_y_train)


# In[ ]:


ffsnn = FFSNNetwork(6, [2, 3])
ffsnn.fit(converted_filtered_train_data, converted_y_train, epochs=1500, learning_rate=.01, display_loss=True)


# In[ ]:


converted_filtered_test_data=np.array(filtered_test_data)


# In[ ]:


Y_pred_val = ffsnn.predict(converted_filtered_test_data)
Y_pred_binarised_val = (Y_pred_val >= 0.5).astype("int").ravel()


# In[ ]:


predicted_y=pd.DataFrame({'PassengerId':filtered_test_data.index,'Survived':Y_pred_binarised_val})


# In[ ]:


predicted_y.to_csv('submission.csv',index=False)


# In[ ]:




