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
df = pd.read_csv('/kaggle/input/titanic/kaggle-titanic-master/kaggle-titanic-master/input/train.csv')
df.head()


# Need to fill NaN values in Age column

# In[ ]:


m = df['Age'].median()
m


# In[ ]:


df['Age'] = df['Age'].fillna(m) 


# In[ ]:


isn = df['Age'].isnull()
isn


# Removing unnecessary columns from DataFrame  

# In[ ]:


df1 = df.drop(['PassengerId','Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1)
df1.head()


# Need to convert Sex column from str to values.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
sex_n = LabelEncoder()
df1['sex_n'] = sex_n.fit_transform(df1['Sex'])
df1.head()


# Removing Sex column from DataFrame

# In[ ]:


df1 = df1.drop('Sex', axis=1)
df1.head()


# data splitting to train and test (using only train.csv data)

# In[ ]:


from sklearn.model_selection import train_test_split


# considering Survived column as target variable.

# In[ ]:


x = df1.drop('Survived', axis = 1)
y = df1['Survived']
# x.head()
# y.head()


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.15, random_state = 10)
# len(x_tran)
# len(x_test)


# Decision Tree model.

# In[ ]:


from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(x_train, y_train)


# Predictions:

# In[ ]:


model.predict(x_test)


# Qualitity test of model: if the socre is more than 0.75 then its good model.

# In[ ]:


model.score(x_test, y_test)


# Fell free to comment with suggestions or any clarifications for further active discussions. 
# My e-mailid is mcommahesh@gmail.com
# 
