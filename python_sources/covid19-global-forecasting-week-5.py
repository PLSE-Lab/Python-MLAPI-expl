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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')
train.head()


# In[ ]:


train['Date'] = pd.to_datetime(train.Date)
train['Month'] = train.Date.dt.month
train['Day'] = train.Date.dt.day
train.head()


# In[ ]:


label1 = ['Country_Region','Population','Weight','Target','Month','Day','TargetValue']
train = train.reindex(labels=label1,axis=1)
train.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelencoder1 = LabelEncoder()
labelencoder2 = LabelEncoder()


# In[ ]:


train['Country_Region'] = labelencoder1.fit_transform(train['Country_Region'])
train['Target'] = labelencoder2.fit_transform(train['Target'])
print(train.head())
train.isnull().sum()
train.info()


# In[ ]:


x_train = train.iloc[:,0:6].values
y_train = train.iloc[:,-1].values


# In[ ]:


test = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')
test.head()


# In[ ]:


test['Date'] = pd.to_datetime(test.Date)
test['Month'] = test.Date.dt.month
test['Day'] = test.Date.dt.day
test.head()


# In[ ]:


label2 = ['Country_Region','Population','Weight','Target','Month','Day']
test = test.reindex(labels=label2,axis=1)
test.head()


# In[ ]:


labelencoder3 = LabelEncoder()
labelencoder4 = LabelEncoder()


# In[ ]:


test['Country_Region'] = labelencoder1.fit_transform(test['Country_Region'])
test['Target'] = labelencoder2.fit_transform(test['Target'])
print(test.head())
test.isnull().sum()
test.info()


# In[ ]:


test.head()


# In[ ]:


x_test = test.iloc[:,:].values


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()


# In[ ]:


model.fit(x_train,y_train)


# In[ ]:


predict = model.predict(x_test)


# In[ ]:


test_df = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv',dtype={'ForecastId':str})
test_df.info()


# In[ ]:


sub = pd.DataFrame({'ForecastId_Quantile':test_df['ForecastId'],'TargetValue':predict})


# In[ ]:


sub.head()


# In[ ]:


sub.info()


# In[ ]:


sub.head()


# In[ ]:


sub.info()


# In[ ]:


sub.to_csv('submission.csv',index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




