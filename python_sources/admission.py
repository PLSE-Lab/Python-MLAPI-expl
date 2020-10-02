#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statsmodels.api as sm
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_train = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')
df_test = pd.read_csv('../input/graduate-admissions/Admission_Predict.csv')


# In[ ]:


df_train.tail(5)


# In[ ]:


df_test.tail(5)


# In[ ]:


df = pd.concat([df_train,df_test],axis=0)


# In[ ]:


df.sample(5)


# In[ ]:


df1 = df.drop('Serial No.',axis=1)


# In[ ]:


df1.head(5)


# In[ ]:


df.info()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = df.iloc[:,:-1]
y = df.iloc[:,-1]


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


reg = LinearRegression()


# In[ ]:


reg.fit(X_train,y_train)


# In[ ]:


pred = reg.predict(X_test)


# In[ ]:


X1 = sm.add_constant(X)


# In[ ]:


model = sm.OLS(y,X).fit()
model.summary()


# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


print(np.sqrt(mean_squared_error(y_test, pred)))


# In[ ]:




