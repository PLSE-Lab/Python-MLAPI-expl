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


# In[ ]:


from IPython.display import display
pd.options.display.max_columns = None


# In[ ]:


#import packages
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report


# In[ ]:


train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')


# In[ ]:


train['dependency']=train['dependency'].map({'yes':1,'no':0})
train['edjefe']=train['edjefe'].map({'yes':1,'no':0})
train['edjefa']=train['edjefa'].map({'yes':1,'no':0})


# In[ ]:


test['dependency']=test['dependency'].map({'yes':1,'no':0})
test['edjefe']=test['edjefe'].map({'yes':1,'no':0})
test['edjefa']=test['edjefa'].map({'yes':1,'no':0})


# In[ ]:


train.head()


# In[ ]:


train=train.fillna(0)
test=test.fillna(0)


# In[ ]:


X=train.drop(columns=['idhogar']).values[:,1:-1]
Y=train.values[:,-1].astype('int')


# In[ ]:


lb=LabelBinarizer()
Y=lb.fit_transform(Y)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=42)


# In[ ]:


forest=RandomForestClassifier(random_state=100)


# In[ ]:


forest.fit(X_train, y_train)


# In[ ]:


pred=forest.predict(X_test)


# In[ ]:


print(classification_report(y_test,pred))


# In[ ]:


testX=test.drop(columns=['idhogar']).values[:,1:]


# In[ ]:


forest.fit(X,Y)


# In[ ]:


pred=lb.inverse_transform(forest.predict(testX))


# In[ ]:


sub=pd.read_csv('../input/sample_submission.csv')


# In[ ]:


sub['Target']=pred


# In[ ]:


sub.to_csv('sample_submission.csv',index=False)


# In[ ]:




