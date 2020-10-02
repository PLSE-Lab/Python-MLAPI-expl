#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/Admission_Predict.csv')
df.head()


# In[ ]:


df.describe()


# In[ ]:


df = df.drop('Serial No.',axis = 1)
df.head()


# In[ ]:


sns.pairplot(df)


# In[ ]:


g = sns.heatmap(df.corr(),annot=True, cmap="PiYG")
g.figure.set_size_inches([8,8])


# In[ ]:


df['CGPA'].hist()


# In[ ]:


df['TOEFL Score'].hist()


# In[ ]:


df['GRE Score'].hist()


# In[ ]:


sns.boxplot(x='University Rating',y='Chance of Admit ',data=df)


# In[ ]:


X = df.drop('Chance of Admit ', axis=1)
y = df['Chance of Admit ']


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[ ]:


scaled_X = scaler.fit_transform(X)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.33, random_state=101)


# In[ ]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[ ]:


lr.fit(X_train, y_train)


# In[ ]:


scaled_pred = lr.predict(X_test)


# In[ ]:


sns.regplot(x=y_test,y=scaled_pred,fit_reg=False)


# In[ ]:


from sklearn.metrics import mean_squared_error
print("MSE:", mean_squared_error(y_test,scaled_pred))


# In[ ]:


df.columns


# In[ ]:


new_X = df[['GRE Score','TOEFL Score','University Rating','CGPA']]


# In[ ]:


new_scale = scaler.fit_transform(new_X)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(new_scale, y, test_size=0.33, random_state=101)


# In[ ]:


lr.fit(X_train, y_train)


# In[ ]:


new_pred = lr.predict(X_test)


# In[ ]:


sns.regplot(x=y_test,y=new_pred,fit_reg=False)


# In[ ]:


print("MSE:", mean_squared_error(y_test,new_pred))


# In[ ]:


sns.distplot(scaled_pred)
plt.title('With All Columns')


# In[ ]:


sns.distplot(new_pred)
plt.title('GRE Score ,TOEFL Score , University Ratin , CGPA columns')

