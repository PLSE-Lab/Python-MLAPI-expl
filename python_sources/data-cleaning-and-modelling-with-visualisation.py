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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/StudentsPerformance.csv')


# In[ ]:


df.head()


# In[ ]:


df.isna().sum()


# In[ ]:


df.describe()


# In[ ]:


passing_marks = 40


# In[ ]:


df['Maths Status'] = np.where(df['math score']<passing_marks,'F','P')
df['Maths Status'].value_counts()


# In[ ]:


sns.countplot(df['Maths Status'],hue=df['gender'],palette='YlOrRd_r')


# In[ ]:


df['Reading Status'] = np.where(df['reading score']<passing_marks,'F','P')
df['Reading Status'].value_counts()


# In[ ]:


sns.countplot(df['Reading Status'],hue = df['gender'],palette='winter' )


# In[ ]:


df['Writing Status'] = np.where(df['writing score']<passing_marks,'F','P')
df['Writing Status'].value_counts()


# In[ ]:


sns.countplot(df['Writing Status'],hue=df['gender'],palette='bright')


# In[ ]:


df['Total Score'] = (df['math score']+ df['reading score']+ df['writing score'])/3


# In[ ]:


df['Pass Status'] = np.where(df['Total Score']<passing_marks,'F','P')
df['Pass Status'].value_counts()


# In[ ]:


sns.countplot(df['Pass Status'],hue=df['gender'])


# In[ ]:


def grades(pass_status,total_marks):
    if pass_status == 'F':
        return 'F'
    if total_marks >=90:
        return 'A+'
    if total_marks >=80:
        return 'A'
    if total_marks >=70:
        return 'B'
    if total_marks >=60:
        return 'C'
    if total_marks >=50:
        return 'D'
    if total_marks >=40:
        return 'E'


# In[ ]:


df['Total_Grade'] = df.apply(lambda x : grades(x['Pass Status'],x['Total Score']),axis=1)


# In[ ]:


df.head()


# In[ ]:


df['gender'].value_counts()


# In[ ]:


df['race/ethnicity'].value_counts()


# In[ ]:


df['parental level of education'].value_counts()


# In[ ]:


df['lunch'].value_counts()


# In[ ]:


df['test preparation course'].value_counts()


# In[ ]:


df.columns


# In[ ]:


df.drop(['math score', 'reading score','writing score','Maths Status', 'Reading Status', 'Writing Status','Pass Status'],axis = 1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


X = df.drop('Total Score',axis = 1)


# In[ ]:


X = pd.get_dummies(X,drop_first=True)


# In[ ]:


plt.figure(figsize=(20,20))
sns.heatmap(X.corr(),annot=True)


# In[ ]:


X.head()


# In[ ]:


y = df['Total Score']


# In[ ]:


x_train,y_train,x_test,y_test = train_test_split(X,y,test_size = 0.3,random_state=101)


# In[ ]:


linear = LinearRegression()


# In[ ]:


linear.fit(x_train,x_test)


# In[ ]:


predict = linear.predict(y_train)


# In[ ]:


mse_linear = mean_squared_error(y_test,predict)
print(mse_linear)


# In[ ]:


rmse_linear = np.sqrt(mse_linear)
print(rmse_linear)


# In[ ]:


from xgboost import XGBRegressor


# In[ ]:


xgb = XGBRegressor()


# In[ ]:


xgb.fit(x_train,x_test)


# In[ ]:


predict_xgb = xgb.predict(y_train)


# In[ ]:


mse_xgb = mean_squared_error(y_test,predict_xgb)
print(mse_xgb)


# In[ ]:


rmse_xgb = np.sqrt(mse_xgb)
print(rmse_xgb)


# **Please Upvote the kernel if you liked it
# Cheers :)**

# In[ ]:




