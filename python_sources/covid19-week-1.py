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


df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')


# In[ ]:


df.loc[df['Lat'].isnull(),'Lat'] = 12.52
df.loc[df['Long'].isnull(),'Long'] = -70.02


# In[ ]:


df.info()


# In[ ]:


def day(x):
    return x.total_seconds()/(60*60*24)
df['Date'] = pd.to_datetime(df['Date'])
df['Days'] = df['Date'] - pd.to_datetime('2020-01-22')
df['Days'] = df['Days'].apply(day)


# In[ ]:


from sklearn.model_selection import train_test_split
X = df[['Lat','Long','Days']]
Y = df['ConfirmedCases']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y)
Z = df['Fatalities']
X_train,X_test,Z_train,Z_test = train_test_split(X,Z)


# In[ ]:


from sklearn.kernel_ridge import KernelRidge
model = KernelRidge()
model.fit(X_train,Y_train)
model2 = KernelRidge()
model2.fit(X_train,Z_train)
model2.score(X_test,Z_test)


# In[ ]:


df2 = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')


# In[ ]:


df2.loc[df2['Lat'].isnull(),'Lat'] = 12.52
df2.loc[df2['Long'].isnull(),'Long'] = -70.02
df2['Date'] = pd.to_datetime(df2['Date'])
df2['Days'] = df2['Date'] - pd.to_datetime('2020-01-22')
df2['Days'] = df2['Days'].apply(day)


# In[ ]:


df3 = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')


# In[ ]:


print(len(df))
print(len(df2))
print(len(df3))


# In[ ]:


X2 = df2[['Lat','Long','Days']]


# In[ ]:


df3['ConfirmedCases'] = model.predict(X2)
df3['Fatalities'] = model2.predict(X2)


# In[ ]:


df3 = df3.set_index('ForecastId')


# In[ ]:


df3.to_csv('submission.csv')


# In[ ]:


#India Lat = 20.59
#India Long = 78.96
#date 22-03-2020 , 60 days from 22-01-2020
model.predict([[35.86,104.19,60]])


# In[ ]:




