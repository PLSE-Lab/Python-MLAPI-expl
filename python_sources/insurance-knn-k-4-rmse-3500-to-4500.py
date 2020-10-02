#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn import preprocessing

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/insurance.csv")
df.head()


# In[ ]:


df.region.unique()


# In[ ]:


df.info()


# In[ ]:


df.replace({"yes":"1","no":"0"},inplace=True)
df["smoker"] = pd.to_numeric(df["smoker"], errors='coerce')


# In[ ]:


df_column_numeric = df.select_dtypes(include=np.number).columns
df_column_category = df.select_dtypes(exclude=np.number).columns
df_category_onehot = pd.get_dummies(df[df_column_category])
df_final = pd.concat([df_category_onehot,df[df_column_numeric]], axis = 1)
df_final.head(10)


# In[ ]:


df_corr=df_final.corr()
df_cov=df_final.cov()
plt.figure(figsize=(12, 9))
sns.heatmap(df_corr,vmin=-1,vmax=1,center=0,annot=True)


# In[ ]:



sns.pairplot(data=df,
                  x_vars=['age','bmi','smoker'],
                  y_vars=['age','bmi','smoker'],hue='region')


# In[ ]:


sns.scatterplot(x=df['smoker'], y=df['expenses'],hue=df['region'],size=df['age'])


# In[ ]:


sns.scatterplot(x=df['age'], y=df['expenses'],hue=df['smoker'],size=df['bmi'])


# In[ ]:


sns.scatterplot(x=df['bmi'], y=df['expenses'],size=df['age'])


# In[ ]:


x= df_final.drop(["expenses"],axis=1)
y=df_final["expenses"]


# In[ ]:


x = preprocessing.StandardScaler().fit(x).transform(x.astype(float))


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=123)


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
from math import sqrt
from sklearn.metrics import mean_squared_error
rmse_val_test = {} #to store rmse values for different k
rmse_val_train={}

for K in range(25):
    K_value = K+1
    neigh = KNeighborsRegressor(n_neighbors = K_value)
    neigh.fit(X_train, y_train) 
    train_y_pred = neigh.predict(X_train)
    error = sqrt(mean_squared_error(y_train,train_y_pred)) #calculate rmse
    rmse_val_train.update(({K:error})) #store rmse values
    test_y_pred = neigh.predict(X_test)
    error = sqrt(mean_squared_error(y_test,test_y_pred)) #calculate rmse
    rmse_val_test.update(({K:error})) #store rmse values


# In[ ]:


elbow_curve_train = pd.Series(rmse_val_train,index=rmse_val_train.keys())
elbow_curve_test = pd.Series(rmse_val_test,index=rmse_val_test.keys())
elbow_curve_train.head(10)


# In[ ]:


k_range = list(range(25))
elbow_curve_train = pd.Series(rmse_val_train,k_range)
elbow_curve_test = pd.Series(rmse_val_test,k_range)
ax=elbow_curve_train.plot(title="RMSE of train VS Value of K ")
ax.set_xlabel("K")
ax.set_ylabel("RMSE of train")


# In[ ]:


ax=elbow_curve_test.plot(title="RMSE of test VS Value of K ")
ax.set_xlabel("K")
ax.set_ylabel("RMSE of test")


# In[ ]:


from sklearn.model_selection import GridSearchCV
k_range = list(range(1, 25))
param_grid = dict(n_neighbors=k_range)
knn = KNeighborsRegressor(param_grid)
model = GridSearchCV(knn, param_grid, cv=5)
model.fit(X_train,y_train)
model.best_params_

