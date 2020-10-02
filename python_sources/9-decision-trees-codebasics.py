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


# In[ ]:


df = pd.read_csv("/kaggle/input/titanic/titanic_train.csv")
df.head()


# In[ ]:


inputs = df[['Pclass', 'Sex', 'Age', 'Fare']]


# In[ ]:


target = df[['Survived']]


# In[ ]:



from sklearn.preprocessing import LabelEncoder
le_sex = LabelEncoder()
inputs['Sex_n'] = le_sex.fit_transform(inputs['Sex'])


# In[ ]:


inputs.drop(['Sex'], axis = 'columns', inplace = True)


# In[ ]:


inputs.Age = inputs.Age.fillna(inputs.Age.mean())


# In[ ]:


from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(inputs, target)


# In[ ]:


model.score(inputs, target)


# 

# 

# In[ ]:




