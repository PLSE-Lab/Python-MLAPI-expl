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
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import RobustScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from keras import Sequential
from keras import layers
from keras import losses
from keras import activations
from keras import optimizers
from keras import backend
from keras import regularizers
import tensorflow as tf


# In[ ]:


def load(fileName):
    return pickle.load(open(fileName,'rb'))
def save(item,fileName):
    pickle.dump(item,open(fileName,'wb'))


# In[ ]:


datai = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")
datat = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")
data = pd.merge(datat,datai,left_on='TransactionID',right_on='TransactionID',how='left')


# In[ ]:


x = data.drop(['isFraud','TransactionID','TransactionDT'],axis=1)
y = data['isFraud']
del data


# In[ ]:


colsNotnum =  x.dtypes.reset_index()
colsNotnum = colsNotnum[colsNotnum[0]=='object']['index']
x[colsNotnum] = x[colsNotnum].fillna(21225)
colsNum = x.dtypes.reset_index()
colsNum = colsNum[colsNum[0]!='object']['index']
x[colsNum] = x[colsNum].fillna(x[colsNum].mean())


# In[ ]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y)


# In[ ]:


lbd = load('/kaggle/input/dataforcisfd/lbdV1')
for col in colsNotnum:
    print(col)
    xtrain[col] = lbd[col].transform(xtrain[col].values.astype(str))
    xtest[col] = lbd[col].transform(xtest[col].values.astype(str))


# In[ ]:


del lbd,colsNotnum,col,colsNum,x,y


# In[ ]:


smt = SMOTE()
xtrain,ytrain = smt.fit_sample(xtrain,ytrain)


# In[ ]:


pd.DataFrame(ytrain).hist()


# In[ ]:


rs = load('/kaggle/input/dataforcisfd/rsV1')
xtrain = rs.transform(xtrain)
xtest = rs.transform(xtest)


# In[ ]:


l1=0.0000001
l2=0.0000001
dropout=0.05


# In[ ]:


with tf.device('GPU:0'):
        
    model = Sequential()
    model.add(layers.Dense(100,activation=activations.linear, 
                           input_dim = xtrain.shape[1],activity_regularizer=regularizers.l1(l1),
                           kernel_regularizer=regularizers.l2(l2)))
    model.add(layers.Activation(activation=activations.relu))
    model.add(layers.normalization.BatchNormalization())
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(100,activation=activations.linear,activity_regularizer=regularizers.l1(l1),
                           kernel_regularizer=regularizers.l2(l2)))
    model.add(layers.Activation(activation=activations.relu))
    model.add(layers.normalization.BatchNormalization())
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(100,activation=activations.linear,activity_regularizer=regularizers.l1(l1),
                           kernel_regularizer=regularizers.l2(l2)))
    model.add(layers.Activation(activation=activations.relu))
    model.add(layers.normalization.BatchNormalization())
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(50,activation=activations.linear,activity_regularizer=regularizers.l1(l1),
                           kernel_regularizer=regularizers.l2(l2)))
    model.add(layers.Activation(activation=activations.relu))
    model.add(layers.normalization.BatchNormalization())
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1,activation=activations.sigmoid))
    model.compile(optimizer=optimizers.Adam(),loss=losses.logcosh)


# In[ ]:


model.fit(xtrain,ytrain,validation_data=(xtest,ytest.values),batch_size=1000,epochs=50,shuffle=True)


# In[ ]:


ypredtrain = model.predict(xtrain)
rocTrain = roc_auc_score(ytrain,ypredtrain)
print(rocTrain)
ypred = model.predict(xtest)
roc = roc_auc_score(ytest,ypred)
print(roc)


# In[ ]:


del xtrain,xtest,ytrain,ytest


# In[ ]:


datai = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv")
datat = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv")
data = pd.merge(datat,datai,left_on='TransactionID',right_on='TransactionID',how='left')
subf = pd.DataFrame(data['TransactionID'])
x = data.drop(['TransactionID','TransactionDT'],axis=1)
colsNotnum =  x.dtypes.reset_index()
colsNotnum = colsNotnum[colsNotnum[0]=='object']['index']
x[colsNotnum] = x[colsNotnum].fillna(21225)
colsNum = x.dtypes.reset_index()
colsNum = colsNum[colsNum[0]!='object']['index']
x[colsNum] = x[colsNum].fillna(0)

lbd = load('/kaggle/input/dataforcisfd/lbdV1')
for col in colsNotnum:
    print(col)
    x[col] = lbd[col].transform(x[col].values.astype(str))
del lbd,colsNotnum,col,colsNum
rs = load('/kaggle/input/dataforcisfd/rsV1')
x = rs.transform(x)
ypred = model.predict(x)
subf['isFraud'] = ypred
subf.to_csv('submission.csv',index=False)

