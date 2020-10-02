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


import matplotlib.pyplot as plt
import seaborn as sns


df_train = pd.read_csv("/kaggle/input/video-game-sales-prediction-machine-hack/Train.csv")
df_test = pd.read_csv("/kaggle/input/video-game-sales-prediction-machine-hack/Test.csv")


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


print(df_train.info())
print(df_test.info())


# In[ ]:


print(df_train["CONSOLE"].unique().sort() == df_test["CONSOLE"].unique().sort())
print(df_train["CATEGORY"].unique().sort() == df_test["CONSOLE"].unique().sort())
print(df_train["PUBLISHER"].unique().sort() == df_test["PUBLISHER"].unique().sort())
print(df_train["RATING"].unique().sort() == df_test["RATING"].unique().sort())


# In[ ]:


numeric_test = df_test.select_dtypes(exclude=["object"]).drop("ID",axis=1)
numeric_test.head()


# In[ ]:


numeric_train = df_train.select_dtypes(exclude=["object"]).drop("ID",axis=1)
numeric_train.head()


# In[ ]:


object_train = df_train.select_dtypes(include=["object"]) #.drop("PUBLISHER",axis=1)
object_train.head()


# In[ ]:


object_test = df_test.select_dtypes(include=["object"]) #.drop("PUBLISHER",axis=1)
object_test.head()


# In[ ]:


object_dummies_tr = pd.get_dummies(object_train)
object_dummies_tr = object_dummies_tr.sort_index(axis=1)




# In[ ]:


object_dummies_te = pd.get_dummies(object_test)

object_dummies_te = object_dummies_te.sort_index(axis=1)




a = []
for i in object_dummies_tr.columns:
    if i in object_dummies_te.columns:
        a.append(i)
        
b = []
for i in object_dummies_te.columns:
    if i in object_dummies_tr.columns:
        b.append(i)
display(a==b)

object_dummies_tr = object_dummies_tr[a]
object_dummies_te = object_dummies_te[a]


# In[ ]:





# In[ ]:


print(object_dummies_tr.shape, object_dummies_te.shape)


# In[ ]:


df_tr = pd.concat([numeric_train,object_dummies_tr],axis=1)

df_tr.shape


# In[ ]:


df_te = pd.concat([numeric_test,object_dummies_te],axis=1)

df_te.shape


# In[ ]:


y_train = df_tr["SalesInMillions"]
X_train = df_tr.drop("SalesInMillions",axis=1)


# In[ ]:


X_test = df_te

X_train.shape,X_test.shape


# In[ ]:


from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import cross_val_score
from sklearn import metrics

lin_reg = LinearRegression()
ridge_reg = Ridge(alpha=35)
lasso_reg = Lasso(alpha=0.0005,max_iter=10000)

lin_reg.fit(X_train,y_train)
ridge_reg.fit(X_train,y_train)
lasso_reg.fit(X_train,y_train)

print(cross_val_score(lin_reg,X_train,y_train,cv=5,scoring="neg_mean_squared_error").mean())
print(cross_val_score(ridge_reg,X_train,y_train,cv=5,scoring="neg_mean_squared_error").mean())
print(cross_val_score(lasso_reg,X_train,y_train,cv=5,scoring="neg_mean_squared_error").mean())



# In[ ]:


from xgboost import XGBRegressor
xg_reg = XGBRegressor()
xg_reg.fit(X_train,y_train)


# In[ ]:


print(cross_val_score(xg_reg,X_train,y_train,cv=5,scoring="neg_mean_squared_error").mean())


# In[ ]:


y_pred = xg_reg.predict(X_test)

print(y_pred)


# In[ ]:


output = pd.DataFrame({"SalesInMillions":y_pred})

output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")

