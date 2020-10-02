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


data=pd.read_csv("../input/social-network-ads/Social_Network_Ads.csv")


# In[ ]:


data.head()


# In[ ]:


data.isnull().sum()


# In[ ]:


from sklearn import preprocessing


# In[ ]:





# In[ ]:


data.head()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


data=pd.get_dummies(data,columns=["Gender"])   # one hot encoding


# In[ ]:


x=data.drop(["User ID","Purchased"],axis=1)
y=data["Purchased"]


# In[ ]:


x.head()


# In[ ]:


y.head()


# In[ ]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.30,random_state=0)     # 0.30 or 30% is the size of test


# In[ ]:


xtrain.head()


# In[ ]:


xtrain["Age"]=pd.cut(xtrain["Age"],bins=3,labels=False)


# In[ ]:


xtrain.head()


# In[ ]:


xtrain["EstimatedSalary"]=pd.cut(xtrain["EstimatedSalary"],bins=10,labels=False)


# In[ ]:


xtrain.head()


# In[ ]:


xtrain.head()


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


lg=LogisticRegression()


# In[ ]:


lg.fit(xtrain,ytrain)


# In[ ]:


xtest["Age"]=pd.cut(xtest["Age"],bins=3,labels=False)


# In[ ]:


xtest["EstimatedSalary"]=pd.cut(xtest["EstimatedSalary"],bins=10,labels=False)


# In[ ]:


xtest.reset_index(inplace=True)


# In[ ]:


xtrain.head()


# In[ ]:


xtest.head()


# In[ ]:


ypred=lg.predict(xtest.drop("index",axis=1))


# In[ ]:


ypred


# In[ ]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[ ]:


confusion_matrix(ytest,ypred)


# In[ ]:


accuracy_score(ytest,ypred)


# In[ ]:


# Accuracy obtained is 83 % 


# In[ ]:




