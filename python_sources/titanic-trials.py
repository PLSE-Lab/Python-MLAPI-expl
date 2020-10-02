#!/usr/bin/env python
# coding: utf-8

# In[213]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[214]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[215]:


train.head()


# In[216]:


test.head()


# In[217]:


train.describe()


# In[218]:


train_title = [i.split(',')[1].split('.')[0].strip() for i in train['Name']]
np.unique(train_title)


# In[219]:


train['Title'] = pd.Series(train_title)
test_title = [i.split(',')[1].split('.')[0].strip() for i in test['Name']]
test['Title'] = pd.Series(test_title)


# In[220]:


train['Title'] = train['Title'].replace(['Lady', 'the Countess', 'Countess', 'Dona', 'Ms', 'Mme', 'Mlle','Miss','Mrs'], 'Ms')
test['Title'] = test['Title'].replace(['Lady', 'the Countess', 'Countess', 'Dona', 'Ms', 'Mme', 'Mlle','Miss','Mrs'], 'Ms')


# In[221]:


train['Title'] = train['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer','Master'], 'Mr')
test['Title'] = test['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer','Master'], 'Mr')


# In[222]:


train['Family_Size'] = train['SibSp'] + train['Parch'] + 1
test['Family_Size'] = test['SibSp'] + test['Parch'] + 1


# In[223]:


def family_size(x):
    if x == 1:
        return 1
    elif x > 1 and x <= 5:
        return 2
    else:
        return 3
    


# In[224]:


train['Family_Size'] = train['Family_Size'].apply(family_size)
test['Family_Size'] = test['Family_Size'].apply(family_size)


# In[225]:


train['Embarked'].fillna(train['Embarked'].mode(),inplace=True)
test['Embarked'].fillna(test['Embarked'].mode(),inplace=True)
train['Age'].fillna(train['Age'].mean(),inplace=True)
test['Age'].fillna(test['Age'].mean(),inplace=True)
train['Fare'].fillna(train['Fare'].mean(),inplace=True)
test['Fare'].fillna(test['Fare'].mean(),inplace=True)


# In[226]:


print(train.head())


# In[227]:


train.info()


# In[228]:


train.shape


# In[229]:


train = train.drop(['Cabin','SibSp','Parch','Ticket','Name'],axis=1)
test_X = test['PassengerId']
test = test.drop(['Cabin','SibSp','Parch','Ticket','Name'],axis=1)


# In[230]:


train['Title'] = train['Title'].replace(['Mr'], 1)
test['Title'] = test['Title'].replace(['Mr'], 1)
train['Title'] = train['Title'].replace(['Ms'], 0)
test['Title'] = test['Title'].replace(['Ms'], 0)


# In[231]:


train['Sex'] = train['Sex'].replace(['male'], 1)
test['Sex'] = test['Sex'].replace(['male'], 1)
train['Sex'] = train['Sex'].replace(['female'], 0)
test['Sex'] = test['Sex'].replace(['female'], 0)


# In[232]:


train['Embarked'] = train['Embarked'].replace(['S'], 1)
test['Embarked'] = test['Embarked'].replace(['S'], 1)
train['Embarked'] = train['Embarked'].replace(['C'], 2)
test['Embarked'] = test['Embarked'].replace(['C'], 2)
train['Embarked'] = train['Embarked'].replace(['Q'], 3)
test['Embarked'] = test['Embarked'].replace(['Q'], 3)


# In[233]:


train = train.dropna()


# In[234]:


test_X = test['PassengerId']
train = train.drop(['PassengerId'],axis = 1)
Y_train = train['Survived']
X_train = train.drop(['Survived'],axis = 1)
test = test.drop(['PassengerId'],axis = 1)


# In[235]:


from sklearn.preprocessing import OneHotEncoder

OHE = OneHotEncoder(categorical_features = [0, 1, 4, 5, 6])
X_train = OHE.fit_transform(X_train).toarray()
test = OHE.fit_transform(test).toarray()


# In[236]:


from sklearn.model_selection import train_test_split


x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size = 0.2)


# In[237]:


x_train.shape


# In[238]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.l1 = nn.Linear(15,30)
        self.l2 = nn.Linear(30,90)
        self.l3 = nn.Linear(90,2)
    
    def forward(self,x):
        out = F.relu(F.dropout(self.l1(x),p=0.2))
        out = F.relu(F.dropout(self.l2(out),p=0.2))
        out = F.sigmoid(self.l3(out))
        
        return out     


# In[239]:


net = Network()

params = list(net.parameters())
print(params)


# In[240]:


batch_size = 50
num_epochs = 50
learning_rate = 0.01
batch_no = len(x_train) // batch_size


# In[241]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


# In[242]:


from sklearn.utils import shuffle
from torch.autograd import Variable

for epoch in range(num_epochs):
    if epoch % 5 == 0:
        print('Epoch {}'.format(epoch+1))
    x_var = Variable(torch.FloatTensor(x_train))
    y_var = Variable(torch.LongTensor(y_train.values))
    # Forward + Backward + Optimize
    optimizer.zero_grad()
    ypred_var = net(x_var)
    loss =criterion(ypred_var, y_var)
    loss.backward()
    optimizer.step()


# In[243]:


test_var = Variable(torch.FloatTensor(x_val), volatile=True)
result = net(test_var)
values, labels = torch.max(result, 1)
num_right = np.sum(labels.data.numpy() == y_val)
print('Accuracy {:.2f}'.format(num_right / len(y_val)))


# In[245]:


test_var = Variable(torch.FloatTensor(test), volatile=True) 
test_result = net(test_var)
values, labels = torch.max(test_result, 1)
survived = labels.data.numpy()


# In[247]:



import csv
submission = [['PassengerId', 'Survived']]
for i in range(len(survived)):
    submission.append([test_X[i], survived[i]])


# In[248]:


with open('submission.csv', 'w') as submissionFile:
    writer = csv.writer(submissionFile)
    writer.writerows(submission)


# In[ ]:




