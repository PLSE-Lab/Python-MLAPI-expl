#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_train.head()


# In[ ]:


df_train = df_train[['Survived', 'Pclass', 'Sex']]
df_train.head()


# In[ ]:


df_train['Sex'] = df_train['Sex'].map({'female':0, 'male':1}).astype(np.uint8)


# In[ ]:


df_train = pd.concat([df_train, pd.get_dummies(df_train['Pclass'], prefix='Pclass')],axis=1)
df_train.drop(['Pclass'],axis=1, inplace=True)


# In[ ]:


df_train.head()


# In[ ]:


X_train = df_train.drop('Survived', axis=1).values
y_train = df_train['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)


# In[ ]:


XTrain, yTrain, XTest, yTest = map(torch.tensor, (X_train, y_train, X_test, y_test))


# In[ ]:


print(XTrain.shape, yTrain.shape)


# In[ ]:


class Model(torch.nn.Module):
    def __init__(self, n_neurons):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(4, n_neurons)
        self.ac1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_neurons, 2) 
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.ac1(x)
        x = self.fc2(x)
        return x
    
model = Model(50)


# In[ ]:


loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)


# In[ ]:


XTrain = XTrain.float()
XTest = XTest.float()


# In[ ]:


for epoch in range(5):
    optimizer.zero_grad()
    
    preds = model(XTrain)
    loss_value = loss(preds, yTrain)
    loss_value.backward()        
    optimizer.step()

    test_preds = model.forward(XTest)        
    accuracy = (test_preds.argmax(dim=1) == yTest).float().mean()    
    
    print(accuracy)


# In[ ]:


df_test = pd.read_csv('../input/test.csv')
df_test.head()


# In[ ]:


df_test = df_test[['Sex','Pclass']]


# In[ ]:


df_test['Sex'] = df_test['Sex'].map({'female':0, 'male':1})


# In[ ]:


df_test = pd.concat([df_test, pd.get_dummies(df_test['Pclass'], prefix='Pclass')],axis=1)
df_test.drop(['Pclass'],axis=1, inplace=True)

df_test.head()


# In[ ]:


pred = torch.from_numpy(df_test.values)


# In[ ]:


sample_sub = pd.read_csv('../input/gender_submission.csv')
sample_sub.head()


# In[ ]:


sample_sub['Survived'] = model(pred.float()).argmax(dim=1)
sample_sub.head()


# In[ ]:


sample_sub.to_csv("submission.csv", index=False)

