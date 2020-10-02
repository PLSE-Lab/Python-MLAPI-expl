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


# ****Loading the Data

# In[ ]:


train=pd.read_csv('/kaggle/input/learn-together/train.csv')
test=pd.read_csv('/kaggle/input/learn-together/test.csv' )
sub=pd.read_csv('/kaggle/input/learn-together/sample_submission.csv')


# Let's see what we have as Data

# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


sub.head()


# What we have as features take a look below

# In[ ]:


train.columns


# Let's see if we have a NaN in our data

# In[ ]:


train.isna().sum().sum()


# The next step is to  split the data(we will give to the train data 80 percent from our data)

# But before we do let's drop the column of the  Id because we don't need to fit it with the another features and don't forget to store it in a variable 

# In[ ]:


train=train.drop(["Id"], axis=1)
Id=test.Id
test=test.drop(['Id'],axis=1)
from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,Y_train,Y_test=train_test_split(train.drop(['Cover_Type'],axis=1),train['Cover_Type'],test_size=0.2,random_state=111)


# In[ ]:


X_train.shape,X_test.shape,Y_train.shape,Y_test.shape


# Model

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# 150 trees

# In[ ]:


model=RandomForestClassifier(n_estimators=150)


# In[ ]:


model.fit(X_train,Y_train)


# In[ ]:


from sklearn.metrics import accuracy_score,classification_report


# In[ ]:


model.score(X_train,Y_train)


# In[ ]:


pred=model.predict(X_test)
accuracy_score(pred,Y_test)


# also you can calculate the accuracy by this methode

# In[ ]:


model.score(X_test,Y_test)


# In[ ]:


test_pred=model.predict(test)


# In[ ]:


sub.head()


# In[ ]:



output=pd.DataFrame({'ID':Id,'Cover_Type':test_pred})


# In[ ]:


output.head()


# In[ ]:


output.to_csv('submission.csv',index=False )


# In[ ]:





# In[ ]:





# In[ ]:




