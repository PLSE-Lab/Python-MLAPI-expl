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


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#from pandas_summary import DataFrameSummary
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn import metrics


# **Funtion For display data**

# In[ ]:


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)


# **Reading Both train and test files**

# In[ ]:


titanic_df = pd.read_csv("../input/train.csv")
titanic_Testdf = pd.read_csv("../input/test.csv")
df_raw = pd.concat([titanic_df,titanic_Testdf])
titanic_df.shape,titanic_Testdf.shape,df_raw.shape
display_all(df_raw.describe(include='all').T)
df_raw.tail(10)
df_raw.head(10)


# **Categorize the SEX column**

# In[ ]:


# convert to category dtype
df_raw['Sex'] = df_raw['Sex'].astype('category')
# convert to category codes
df_raw['Sex'] = df_raw['Sex'].cat.codes
df_raw


# In[ ]:


# subset all categorical variables which need to be encoded doing hot encoding on  Embarked column
cat = ['Embarked']
for var in cat:
 df_raw = pd.concat([df_raw,pd.get_dummies(df_raw[var],prefix=var)],axis=1)
del df_raw[var]
df_raw
   
                       


# dropping few columns

# In[ ]:


df_raw.drop(['Name','Ticket','PassengerId','Cabin'],axis=1,inplace=True)
df_raw.head(400)


# In[ ]:


df_raw.isnull().any()
#Replaced Nan values in Age column with it's Mean
df_raw.ix[:,0]=(df_raw.ix[:,0]).fillna(df_raw.mean(axis=0)[0])
df_raw.ix[:,1]=(df_raw.ix[:,1]).fillna(df_raw.mean(axis=0)[1])
df_raw.isnull().any()


# In[ ]:


df_titanic = df_raw[pd.notnull(df_raw['Survived'])]
df_testtitanic = df_raw[pd.isnull(df_raw['Survived'])]
df_testtitanic.drop(['Survived'],axis=1,inplace=True)
df_titanic.isnull().any()
df_testtitanic.isnull().any()


# In[ ]:


#Validation data
X_train, X_val, y_train, y_val = train_test_split(
    df_titanic.drop(['Survived'], axis=1),
    df_titanic['Survived'],
    test_size=0.2, random_state=42)


# In[ ]:


for i in (X_train,X_val,y_train,y_val):
    print(i.shape)
    


# Creating Random Forest  instance model with Training data

# In[ ]:


m = RandomForestClassifier(n_estimators=10,random_state=42)
m.fit(X_train,y_train)


# Since result is in categorical variable like survived or Not Survived and it is not uniform like 75%  of our data is not survived so our model accuracy will not come correct if we use R2 accuracy paramaeter.
# 
# F1 score is right accuracy measure for such kind of dataset.

# In[ ]:


#accuracy_score(y_train, m.predict(X_val))
#m.score(X_train,y_train)
#m.score(X_val,y_val)
#accuracy_score(y_train,m.predict(X_train))
#accuracy_score(y_val, m.predict(X_val))
f1_score(y_val, m.predict(X_val))
#quetions what if i have 3 categoricalvariables like 0,1,2 then how f1 score will works ??
#m.predict(df_testtitanic)


# So as per above model we are getting 75% accuracy on validation data set.

# In[ ]:



titanic_Testdf['Survived'] = m.predict(df_testtitanic)
titanic_Testdf


# In[ ]:


solution = titanic_Testdf[['PassengerId','Survived']]
solution['Survived'] = solution['Survived'].apply(int)


# In[ ]:


solution.to_csv("Random_Forest_Solution.csv", index=False)

