#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data manipulation
import numpy as np
import pandas as pd

# data visualisation
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from sklearn import metrics

# sets matplotlib to inline
get_ipython().run_line_magic('matplotlib', 'inline')

# importing LogisticRegression for Test and Train
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[ ]:


df = pd.read_csv("../input/HR-Employee-Attrition.csv")


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.columns


# In[ ]:


df['Attrition'] = df['Attrition'].map(lambda x: 1 if x== 'Yes' else 0)


# In[ ]:


df.head()


# In[ ]:


df.dtypes


# In[ ]:


df.isnull().any()


# In[ ]:


df.corr()


# In[ ]:


def plot_factorplot(attr,labels=None):
    sns.catplot(data=df,kind='count',height=5,aspect=1.5,x=attr)


# In[ ]:


cat_df=df.select_dtypes(include='object')

for i in cat_df:
    plt.figure(figsize=(15, 15))
    plot_factorplot(i)   


# In[ ]:


df.corr()


# In[ ]:


df.drop(labels=['EmployeeCount','EmployeeNumber','StockOptionLevel','StandardHours'],axis=1,inplace=True)
df.head()


# In[ ]:


df.corr()


# In[ ]:


df.cov()


# In[ ]:


#cat_col = df.select_dtypes(exclude=np.number).columns
cat_col = df.select_dtypes(exclude=np.number)
cat_col


# #finding value_counts on all categorical variable

# In[ ]:


for i in cat_col:
    print(df[i].value_counts())


# In[ ]:


numerical_col = df.select_dtypes(include=np.number)
numerical_col


# In[ ]:


for i in numerical_col:
    print(i)


# In[ ]:


df.BusinessTravel.value_counts()


# **ENCODING**

# In[ ]:


df.columns.shape


# In[ ]:


one_hot_categorical_variables = pd.get_dummies(cat_col)


# In[ ]:


one_hot_categorical_variables.head()


# In[ ]:


df = pd.concat([numerical_col,one_hot_categorical_variables],sort=False,axis=1)
df.head()


# Now lets set the target variable and remove from the actual DataFrame

# In[ ]:


x = df.drop(columns='Attrition')


# In[ ]:


y = df['Attrition']


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=12)
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
train_Pred = logreg.predict(x_train)


# In[ ]:


metrics.confusion_matrix(y_train,train_Pred)


# In[ ]:


metrics.accuracy_score(y_train,train_Pred)


# In[ ]:


test_Pred = logreg.predict(x_test)


# In[ ]:


metrics.confusion_matrix(y_test,test_Pred)


# In[ ]:


metrics.accuracy_score(y_test,test_Pred)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, test_Pred))


# In[ ]:





# In[ ]:




