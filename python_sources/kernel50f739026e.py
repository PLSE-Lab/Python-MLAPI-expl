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


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


df_iris = pd.read_csv('../input/iris/Iris.csv')


# In[ ]:


df_iris.describe()


# In[ ]:


df_iris.head()


# In[ ]:


df_iris.columns


# In[ ]:


df_iris.shape


# In[ ]:


df_iris.isnull().sum()


# In[ ]:


df_iris.select_dtypes('object').head()


# In[ ]:


y = df_iris.Species


# In[ ]:


y.head()


# In[ ]:


y = pd.get_dummies(y)


# In[ ]:


y.head()


# In[ ]:


X = df_iris.drop(columns = ['Species'],axis = 0)


# In[ ]:


X.head()


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)


# In[ ]:


X_train.head()


# In[ ]:


model_RF = RandomForestRegressor(random_state=1)


# In[ ]:


model_RF.fit(X_train, y_train)


# In[ ]:


predictions = model_RF.predict(X_val)


# In[ ]:


predictions


# In[ ]:


mae = mean_absolute_error(y_val, predictions)


# In[ ]:


print(" MAE score : ", mae)


# In[ ]:




