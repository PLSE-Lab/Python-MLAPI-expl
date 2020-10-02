#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data=pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')


# In[ ]:


data


# In[ ]:


data=data[data['status']=='Placed']


# In[ ]:


data


# In[ ]:


data.info()


# In[ ]:


data.describe(include='all')


# In[ ]:


new_data=data[data['salary']<=600000]


# In[ ]:


X=data.drop(columns=['sl_no','status','salary'])


# In[ ]:


y=data['salary']


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2)


# In[ ]:


ohe=OneHotEncoder()
ohe.fit(X[['gender','ssc_b','hsc_b','hsc_s','degree_t','workex','specialisation']])


# In[ ]:


ohe.categories_


# In[ ]:


col_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['gender','ssc_b','hsc_b','hsc_s','degree_t','workex','specialisation']),
                                 remainder='passthrough')


# In[ ]:


lr=LinearRegression()


# In[ ]:


pipe=make_pipeline(col_trans,lr)


# In[ ]:


pipe.fit(X_train,y_train)


# In[ ]:


y_pred=pipe.predict(X_test)


# In[ ]:


r2_score(y_pred,y_test)


# In[ ]:


y_test


# In[ ]:


y_pred


# ## Analysis with Target Column
# 
# ### Dependence of Gender with salary

# In[ ]:


sns.boxplot(x='gender',y='salary',data=new_data)


# ### Dependence of ssc_p with salary

# In[ ]:


sns.relplot(x='ssc_p',y='salary',data=new_data)


# ## Dependence of ssc_b with salary

# In[ ]:


sns.boxplot(x='ssc_b',y='salary',data=new_data)


# ## Dependence of hsc_p with salary

# In[ ]:


sns.relplot(x='ssc_p',y='salary',hue='hsc_b',style='hsc_s',data=new_data)


# ## Dependence of degree_p with salary

# In[ ]:


sns.relplot(x='degree_p',y='salary',hue='degree_t',style='workex',data=new_data)


# 
# ## Dependence of etest_p with salary

# In[ ]:


sns.relplot(x='etest_p',y='salary',hue='specialisation',data=new_data)


# ## Dependence of mba_p with salary

# In[ ]:


sns.relplot(x='mba_p',y='salary',hue='specialisation',data=new_data)


# In[ ]:


X=new_data.drop(columns=['sl_no','status','salary','ssc_b'])
y=new_data['salary']


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder,PolynomialFeatures
from sklearn.metrics import r2_score


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2)


# In[ ]:


ohe=OneHotEncoder()
ohe.fit(X[['gender','hsc_b','hsc_s','degree_t','workex','specialisation']])
ohe.categories_


# In[ ]:


col_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['gender','hsc_b','hsc_s','degree_t','workex','specialisation']),
                                  #(PolynomialFeatures(2),['ssc_p','hsc_p','degree_p','etest_p','mba_p']),
                                 remainder='passthrough')


# In[ ]:


lr=LinearRegression()


# In[ ]:


pipe=make_pipeline(col_trans,lr)


# In[ ]:


pipe.fit(X_train,y_train)


# In[ ]:


y_pred=pipe.predict(X_test)


# In[ ]:


r2_score(y_test,y_pred)


# In[ ]:


r2_score(y_pred,y_test)


# In[ ]:


y_test


# In[ ]:


y_pred


# In[ ]:




