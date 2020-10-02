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


titanic_file_path = '/kaggle/input/titanic/train.csv'


# In[ ]:


titanic_data=pd.read_csv(titanic_file_path)


# In[ ]:


print(titanic_data.columns)


# In[ ]:


y = titanic_data.Survived


# In[ ]:


print(y)


# In[ ]:


titanic_parameters=['PassengerId', 'Pclass', 'Age', 'SibSp', 'Fare']


# In[ ]:


print(titanic_parameters)


# In[ ]:


X = titanic_data[titanic_parameters]


# In[ ]:


X.describe()


# In[ ]:


X.head()


# In[ ]:


#to clean up data and get rid of missing values
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
new_data = my_imputer.fit_transform(X)
print(new_data)


# In[ ]:


#machine learning model
from sklearn.tree import DecisionTreeRegressor
titanic_model = DecisionTreeRegressor(random_state=1)
titanic_model.fit(new_data,y)


# In[ ]:


#to predict with training data set
predictions = titanic_model.predict(new_data)


# In[ ]:


print(predictions)


# In[ ]:


#to predict with test data set
final_test = '/kaggle/input/titanic/test.csv'


# In[ ]:


final_titanic_data=pd.read_csv(titanic_file_path)


# In[ ]:


final_y = final_titanic_data.Survived


# In[ ]:


final_titanic_parameters=['PassengerId', 'Pclass', 'Age', 'SibSp', 'Fare']


# In[ ]:


final_X = final_titanic_data[final_titanic_parameters]


# In[ ]:


final_X.head()


# In[ ]:


final_new_data = my_imputer.fit_transform(final_X)
print(final_new_data)


# In[ ]:


final_titanic_model = DecisionTreeRegressor(random_state=1)
final_titanic_model.fit(final_new_data,final_y)


# In[ ]:


final_predictions = final_titanic_model.predict(final_new_data)


# In[ ]:


print(final_predictions)


# In[ ]:


print(final_X)

