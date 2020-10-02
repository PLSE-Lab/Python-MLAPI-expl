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


df = pd.read_csv("../input/train_file.csv")
df_test = pd.read_csv("../input/test_file.csv")


# In[ ]:


df.shape


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df.hist(figsize=(12,8))


# In[ ]:


df.columns


# In[ ]:


df.groupby('Sex')['Sex'].count()
df.groupby('Sex')['Greater_Risk_Probability'].mean()


# In[ ]:


df.groupby('Race')['Race'].count()
df.groupby('Race')['Greater_Risk_Probability'].mean()


# In[ ]:


m = df.groupby('LocationDesc')['LocationDesc'].count() < 100


# In[ ]:


cols_drop = []
for i in m.index:
    if m[i] == True:
        cols_drop.append(i)

    


# In[ ]:


def dropX(x):
    if x in cols_drop:
        return 0
    else:
        return 1
    


# In[ ]:





# In[ ]:


df['LocationDesc'] = df['LocationDesc'].apply(dropX, 1)


# In[ ]:


df.shape


# In[ ]:


df.groupby('YEAR')['Patient_ID'].count()


# In[ ]:


df.drop('GeoLocation',axis=1,inplace=True)
df_test.drop('GeoLocation',axis=1,inplace=True)


# In[ ]:


df.shape


# In[ ]:


df.groupby('StratificationType')['Greater_Risk_Probability'].mean()


# In[ ]:


df = pd.get_dummies(df,columns=["StratificationType"])
df_test = pd.get_dummies(df_test,columns=["StratificationType"])


# In[ ]:


df.shape


# In[ ]:


df.drop('Description',axis=1, inplace=True)
df_test.drop('Description',axis=1, inplace=True)


# In[ ]:


df.head(5)


# In[ ]:


df.groupby('Race')['Race'].count()


# In[ ]:


def RaceToken(x):
    if x == 'Asian':
        return 0
    if x== 'Black or African American':
        return 1
    if x == 'Hispanic or Latino':
        return 2
    if x == 'Multiple Race':
        return 3
    if x == 'White':
        return 4
    else:
        return 5
        


# In[ ]:


df['Race'] = df['Race'].apply(RaceToken,1)
df_test['Race'] = df_test['Race'].apply(RaceToken,1)


# In[ ]:


df.head(5)


# In[ ]:


df.groupby('Subtopic')['Subtopic'].count()


# In[ ]:


df.hist(figsize=(20,10))


# In[ ]:


df.corr().hist(figsize = (30,10))


# In[ ]:


def SexClass(x):
    if x == 'Female':
        return 0
    if x == 'Male':
        return 1
    if x == 'Total':
        return 2


# In[ ]:


df['Sex'] = df['Sex'].apply(SexClass, 1)
df_test['Sex'] = df_test['Sex'].apply(SexClass, 1)


# In[ ]:


df.head()


# In[ ]:


from nltk.tokenize import word_tokenize


# In[ ]:


df.drop('YEAR',axis=1,inplace=True)
df_test.drop('YEAR',axis=1,inplace=True)


# In[ ]:


df.groupby('QuestionCode')['QuestionCode'].count()


# In[ ]:


def groupQuestions(x):
    if x == 'QNHALLUCDRUG':
        return 0
    else:
        return x[1:]
    


# In[ ]:


df['drgInt'] = df['QuestionCode'].apply(groupQuestions, 1)
df_test['drgInt'] = df_test['QuestionCode'].apply(groupQuestions, 1)


# In[ ]:


df.drop('QuestionCode',axis =1 , inplace = True)
df_test.drop('QuestionCode',axis =1 , inplace = True)


# In[ ]:


df.columns


# In[ ]:


df.shape
df_test.shape


# In[ ]:


df_test.head()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


df.head()


# In[ ]:


df_test['LocationDesc'] = df_test['LocationDesc'].apply(dropX, 1)


# In[ ]:


df_test.head()


# In[ ]:


df.drop('Greater_Risk_Question',axis=1,inplace=True)
df_test.drop('Greater_Risk_Question',axis=1,inplace=True)


# In[ ]:


y = df["Greater_Risk_Probability"]
X  = df.drop('Greater_Risk_Probability',axis=1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)


# **LOGISTIC REGRESSION USES**

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lr = LinearRegression()
X_train['drgInt'] = pd.to_numeric(X_train['drgInt'])


# In[ ]:


lr.fit(X_train,y_train)


# In[ ]:


y_pred = lr.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


lr.score(X_test,y_test)


# **RANDOM FOREST CLASSIFIER**

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


rr = RandomForestRegressor()


# In[ ]:


rr.fit(X_train,y_train)


# In[ ]:


y_pred = rr.predict(X_test)


# In[ ]:


rr.score(X_test,y_test)


# In[ ]:


from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[ ]:


get_ipython().run_line_magic('pinfo', 'mean_squared_error')


# In[ ]:


mse = mean_squared_error(y_true=y_test,y_pred=y_pred)


# In[ ]:


rmse = np.sqrt(mse)


# In[ ]:


rmse


# In[ ]:


plt.scatter(y_pred,y_test)


# In[ ]:




