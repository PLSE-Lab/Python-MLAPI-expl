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


import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
df_test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")


# In[ ]:


display(df_train.head())
display(df_train.columns)


# In[ ]:


display(df_test.head())

display(df_test.columns)


# In[ ]:


display(df_test.info())


# In[ ]:


fig,ax=plt.subplots(figsize=(30,15))

sns.heatmap(df_train.isnull(),ax=ax)


# In[ ]:


fig,ax2=plt.subplots(figsize=(30,15))
sns.heatmap(df_test.isnull(),ax=ax2)


# In[ ]:


numeric_train = df_train.select_dtypes(exclude=['object']).drop(["SalePrice",'Id'], axis = 1)
display(numeric_train.head())

id_test = df_test["Id"]
y_train = df_train["SalePrice"]

numeric_test = df_test.select_dtypes(exclude=['object']).drop('Id', axis = 1)
display(numeric_test.head())


# In[ ]:


display(numeric_train.shape)
display(numeric_test.shape)

print("original data")
display(df_train.shape)
display(df_test.shape)


# In[ ]:


object_train = df_train.select_dtypes(include=['object'])
object_test = df_test.select_dtypes(include=['object'])

display(object_train.shape)
display(object_test.shape)

unique_1=[]
for col in object_train:
    unique_1.append(object_train[col].nunique())
    
print(unique_1)


# In[ ]:


unique_2=[]
for col in object_test:
    unique_2.append(object_test[col].nunique())
    
print(unique_2)


# In[ ]:


object_index_list = np.array(object_train.columns)[np.array(unique_1) == np.array(unique_2)]

object_train = object_train[object_index_list]
object_test = object_test[object_index_list]


# In[ ]:


object_train.isnull().sum()


# In[ ]:


object_dummies_train = pd.get_dummies(object_train)

object_dummies_test = pd.get_dummies(object_test)

display(object_dummies_train.shape)
display(object_dummies_test.shape)


# In[ ]:


#imputer

import numpy as np
from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer(strategy="mean")
imp_median = SimpleImputer(strategy="median")
imp_mode = SimpleImputer(strategy="most_frequent")

imp_mean.fit(numeric_train)
numeric_train = imp_mean.transform(numeric_train)

imp_mean.fit(numeric_test)
numeric_test = imp_mean.transform(numeric_test)

numeric_train = pd.DataFrame(numeric_train)
numeric_test = pd.DataFrame(numeric_test)


# In[ ]:


X_train = pd.concat([numeric_train,object_dummies_train],axis=1)

display(X_train)

X_test = pd.concat([numeric_test,object_dummies_test],axis=1)

display(X_test)


# In[ ]:


display(X_train.shape)
display(y_train.shape)
display(X_test.shape)


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# In[ ]:


print(X_val.shape)
print(y_val.shape)
print(X_train.shape)


# In[ ]:


get_ipython().system('pip install d2l==0.13.2 -f https://d2l.ai/whl.html # installing d2l')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from d2l import torch as d2l
import torch
import torch.nn as nn

n_train = X_train.shape[0]

train_features = torch.tensor(X_train.values,dtype = torch.float32)
test_features = torch.tensor(X_test.values,dtype = torch.float32)
train_labels = torch.tensor(y_train.values,dtype=torch.float32).reshape(-1, 1)
val_features = torch.tensor(X_val.values,dtype = torch.float32)
val_labels = torch.tensor(y_val.values,dtype=torch.float32).reshape(-1, 1)
# print(val_labels.shape)

loss = nn.MSELoss()
in_features = train_features.shape[1]

# print(in_features)

def get_net():
    net = nn.Sequential(nn.Linear(in_features,1))
    return net


# In[ ]:


def log_rmse(net,features,labels):
    clipped_preds = torch.clamp(net(features),1,float('inf'))
    rsme = torch.sqrt(torch.mean(loss(torch.log(clipped_preds),torch.log(labels))))
    return rsme.item()


# In[ ]:


def train_pros(net,train_features,train_labels,val_features,val_labels,num_epochs,learning_rate,weight_decay):
    
    train_ls, val_ls = [], []
    
    optimizer = torch.optim.Adam(net.parameters(),lr = learning_rate , weight_decay = weight_decay)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = net(train_features)
        l = loss(output , train_labels)
        l.backward()
        optimizer.step()
        train_ls.append(log_rmse(net,train_features,train_labels))
        if val_labels is not None:
            val_ls.append(log_rmse(net, val_features, val_labels))
    return train_ls , val_ls


# In[ ]:


def train(train_features,train_labels,val_features,val_labels, num_epochs,
           learning_rate, weight_decay, batch_size):
    
        train_l_sum, valid_l_sum = 0, 0

        net = get_net()
        train_ls, valid_ls = train_pros(net,train_features, train_labels,val_features,val_labels, num_epochs, learning_rate,
                                   weight_decay)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        d2l.plot(list(range(1, num_epochs+1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse',
                     legend=['train', 'valid'], yscale='log')
        print(' train rmse: %f, valid rmse: %f' % (
             train_ls[-1], valid_ls[-1]))
        return train_l_sum, valid_l_sum


# In[ ]:


num_epochs, lr, weight_decay, batch_size = 100, 5, 0, 64
train_l, valid_l = train(train_features,train_labels,val_features,val_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(' validation: avg train rmse: %f, avg valid rmse: %f'
      % (train_l, valid_l))


# In[ ]:


def pred(test_feature):

    net = get_net()
    preds = net(test_features).detach().numpy()
    
    return preds


# In[ ]:


num_epochs, lr, weight_decay, batch_size = 100, 5, 0, 64
y_predict = pred(test_features)
df_test['SalePrice'] = pd.Series(y_predict.reshape(1, -1)[0])


# In[ ]:


submission = pd.concat([df_test['Id'], df_test['SalePrice']], axis=1)
submission.to_csv('submission.csv', index=False)


# In[ ]:


submission.tail()


# In[ ]:


submission.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")


# In[ ]:




