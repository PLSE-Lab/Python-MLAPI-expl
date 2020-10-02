#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train=pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')


# In[ ]:


train.head()


# In[ ]:


train.describe().T


# In[ ]:


features=["ssc_p","hsc_p"]
X = train[features]


# In[ ]:


X.head()


# In[ ]:



y = (train["salary"])

y.head()


# In[ ]:


train.isnull().sum()


# In[ ]:


cols_with_missing = [col for col in train.columns
                     if train[col].isnull().any()]

print(cols_with_missing)


# In[ ]:


train.dtypes.sample(10)


# In[ ]:


train = pd.get_dummies(train)


# In[ ]:


train.head()


# In[ ]:


train.dtypes.sample(10)


# In[ ]:


from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(train))
imputed_X_train.columns = train.columns


# In[ ]:


imputed_X_train.head()


# In[ ]:


y=imputed_X_train["salary"]


# In[ ]:


y.head()


# In[ ]:


X.head()


# In[ ]:


y.isnull().sum()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[ ]:


import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)


# In[ ]:


y_train.tail()


# In[ ]:


y_test


# In[ ]:


pred=reg.predict(X_test)


# In[ ]:


pred


# In[ ]:


y_test


# In[ ]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': pred})
df


# In[ ]:


from sklearn.metrics import r2_score
print(r2_score(y_train, reg.predict((X_train))))


# In[ ]:


from sklearn.metrics import mean_absolute_error


mean_absolute_error(y_test, pred)


# In[ ]:


output = pd.DataFrame({'sscp':X_test.ssc_p,'hscp':X_test.hsc_p,'Salary': pred})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")


# In[ ]:




