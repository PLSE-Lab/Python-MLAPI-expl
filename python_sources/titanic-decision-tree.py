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


df = pd.read_csv('/kaggle/input/titanicdtree/titanic.csv')
df.head()


# In[ ]:


inputs = df.drop(['PassengerId','Survived','Name','SibSp','Parch','Ticket','Cabin','Embarked'], axis = 'columns')
inputs.head()


# In[ ]:


target = df['Survived']
target.head()


# In[ ]:


inputs.Sex = inputs.Sex.map({'male':1,'female':2})


# In[ ]:


inputs.head()


# In[ ]:


inputs.info()


# In[ ]:


inputs.Age = inputs.Age.fillna(inputs.Age.mean())


# In[ ]:


inputs.Age[:10]


# In[ ]:


inputs.head()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,Y_train,Y_test = train_test_split(inputs, target, test_size = 0.2)


# In[ ]:


print(len(X_train))
print(len(X_test))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[ ]:


model.fit(X_train,Y_train)


# In[ ]:


model.predict(X_test)


# In[ ]:


model.score(X_test,Y_test)

