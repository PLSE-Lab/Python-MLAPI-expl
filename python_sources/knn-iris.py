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


# In[ ]:


def converter(x):
    if x=='Iris-setosa':
        return 0
    if x=='Iris-versicolour':
        return 1
    else:
        return 2 


# In[ ]:


df=pd.read_csv('../input/iris/Iris.csv',converters={'Species':converter})


# In[ ]:


df.head()


# In[ ]:


df.drop(columns='Id',axis=0,inplace=True)


# In[ ]:


df.head()


# In[ ]:


x=df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[ ]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.3, random_state=0)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(xtrain, ytrain)


# In[ ]:


ypred=knn.predict(xtest)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


result=accuracy_score(ytest, ypred, normalize=True, sample_weight=None)


# In[ ]:


result


# In[ ]:




