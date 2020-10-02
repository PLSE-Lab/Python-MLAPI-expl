#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


# In[ ]:


train=pd.read_csv("../input/train.csv",header=0)


# In[ ]:


def maleorfe(var):
    if var=='female':
        return 0
    else:
        return 1


# In[ ]:


train['Sex']=train['Sex'].apply(lambda x:maleorfe(x))


# In[ ]:


df=train[['PassengerId','Pclass','Parch','SibSp','Age','Sex','Survived']]


# In[ ]:


df.dropna(axis=0,how='any',inplace=True)


# In[ ]:


X=df[['PassengerId', 'Pclass', 'Age','Sex']]
y=df['Survived']


# In[ ]:


xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.25)


# In[ ]:


clf=LogisticRegression()
clf.fit(xtrain,ytrain)
pred=clf.predict(xtest)
acc=accuracy_score(pred,ytest)


# In[ ]:


acc


# In[ ]:


test=pd.read_csv("../input/test.csv",header=0)


# In[ ]:


test['Sex']=test['Sex'].apply(lambda x:maleorfe(x))


# In[ ]:


x_test=test[['PassengerId', 'Pclass', 'Age', 'Sex']]


# In[ ]:


x_test['Age'].fillna(0, inplace=True)


# In[ ]:


pr=clf.predict(x_test).astype(int)


# In[ ]:


ans = pd.DataFrame({'PassengerId':x_test.PassengerId, 'Survived':pr})


# In[ ]:


ans.to_csv('finalanswer.csv', index=False)


# In[ ]:


import csv

with open('finalanswer.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        print(row[0],row[1],sep=',')

