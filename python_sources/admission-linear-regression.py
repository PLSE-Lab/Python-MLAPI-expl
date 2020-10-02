#!/usr/bin/env python
# coding: utf-8

# # Problem Attributes
# 
# * GRE Scores ( out of 340 )
# * TOEFL Scores ( out of 120 )
# * University Rating ( out of 5 )
# * Statement of Purpose and Letter of Recommendation Strength ( out of 5 )
# * Undergraduate GPA ( out of 10 )
# * Research Experience ( either 0 or 1 )
# * Chance of Admit ( ranging from 0 to 1 ) -Target Variable

# # Problem Statement
# 
# Going to predict the probability of getting admission.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib as plt
from sklearn.linear_model import LinearRegression, ridge_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


d=open('../input/graduate-admissions/Admission_Predict.csv')
for x in range(5):
    print(d.readline())


# In[ ]:


df=pd.read_csv('../input/graduate-admissions/Admission_Predict.csv',index_col=['Serial No.'],sep=',')
df.head()


# In[ ]:


df.columns


# In[ ]:


df.columns=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA','Research', 'Chance of Admit']


# In[ ]:


y=df['Chance of Admit']
X=df.drop(['Chance of Admit'],axis=1)
X.head()


# In[ ]:


ss=StandardScaler()
X=ss.fit_transform(X)
X=pd.DataFrame(X,columns=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA','Research'])
X


# In[ ]:


df_corr=X.corr()
df_corr


# In[ ]:


sns.heatmap(df_corr)


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=5)
lr=LinearRegression()
lr.fit(X_train,y_train)
y_predict=lr.predict(X_test)

r2_score(y_test,y_predict)

