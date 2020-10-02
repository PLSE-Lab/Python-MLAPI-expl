#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In this notebook, I want to explore some of the basic practices classification works. I have currently committed a basic logistic regression. I plan to explore 
# (1) logistic regression
# (2) svm 
# (3) effects of scaling vs current unscaled data
# (4) effects of introducing balance or weights
# (5) train some neural network later.
# 

# In[ ]:


def produce_accuracy(list1,list2):
    if len(list1)!=len(list2):
        raise ValueError('lengths do not match')
    else:
        no_of_matches=0
        for i in range(len(list1)):
            if list1[i]==list2[i]:
                no_of_matches+=1
        percent=no_of_matches/len(list1)
        return percent


# In[ ]:


data=pd.read_csv('../input/data.csv')
cols=list(data.columns)
print(cols)
data=data.fillna(0)


# In[ ]:


print(data.shape)
print(data.head())


# In[ ]:


print(data.diagnosis.unique().tolist())
print(len(data[data['diagnosis']=='M'].diagnosis))
def binarizer(x):
    if x=='M':
        return 0
    else:
        return 1
data['binary_class']=data['diagnosis'].apply(lambda x:binarizer(x))
data=data.drop(['diagnosis','Unnamed: 32'],axis=1)


# In[ ]:


data=data.drop(['id'],axis=1)
print(data.shape)
print(data.columns)


# In[ ]:


import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.preprocessing import scale
from sklearn.metrics import precision_recall_fscore_support
#create train,test,validation split with 20% fractions
data_train=data.sample(frac=0.8,random_state=200)
data_test=data.drop(data_train.index)
data_validation=data_train.sample(frac=0.1,random_state=210)
data_train_ultimate=data_train.drop(data_validation.index)
X_train=data_train_ultimate.drop(['binary_class'],axis=1)
X_test=data_test.drop(['binary_class'],axis=1)
Y_train=data_train_ultimate['binary_class']
Y_test=data_test['binary_class']
validation_test=data_validation.drop(['binary_class'],axis=1)
validation_actual=data_validation['binary_class']


# In[ ]:


print(data.shape)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
print(validation_test.shape)
print(validation_actual.shape)


# In[ ]:


#let's try logistic regression
#first time with default 100 max_iter it didn't converge. So, increased to 1000.
logistic_model=LogisticRegression(solver='lbfgs',max_iter=1000,n_jobs=-1)
logistic_model.fit(X_train,Y_train)
predictions=logistic_model.predict(X_test)
print('the naive prediction-accuracy is:',produce_accuracy(predictions,Y_test.tolist()))
print('the precision,recall and fscore results are: ',precision_recall_fscore_support(predictions,Y_test.tolist()))
validation_prediction=logistic_model.predict(validation_test)
print('the accuracy in validation set is:',produce_accuracy(validation_actual.tolist(),validation_prediction))
print('the precision,recall and fscore results are: ',precision_recall_fscore_support(validation_prediction,validation_actual.tolist()))


# In[ ]:




