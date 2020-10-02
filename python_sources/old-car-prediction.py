#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv(os.path.join(dirname, filename))
df


# In[ ]:


df.corr()


# In[ ]:


fig = plt.figure(figsize=(10,8))
ax = sns.heatmap(df.corr(),annot=True,cmap = 'viridis')
fig.show()


# In[ ]:


sns.scatterplot(x = 'current price',y = 'km',data = df)


# In[ ]:


sns.scatterplot(x = 'current price',y = 'top speed',data = df)


# In[ ]:


sns.countplot(x = 'condition',data = df)


# In[ ]:


sns.pairplot(df)


# In[ ]:


X = df.drop('current price',axis=1)
y = df['current price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)


# In[ ]:


lm = LinearRegression()
lm.fit(X_train,y_train)
pred_linear = lm.predict(X_test)
print(r2_score(pred_linear,y_test))
print(mean_absolute_error(pred_linear,y_test))
print(mean_squared_error(pred_linear,y_test))


# In[ ]:


plt.scatter(pred_linear,y_test)


# In[ ]:


rtree = RandomForestRegressor()
rtree.fit(X_train,y_train)
pred_tree = rtree.predict(X_test)
print(r2_score(pred_tree,y_test))
print(mean_absolute_error(pred_tree,y_test))
print(mean_squared_error(pred_tree,y_test))


# In[ ]:


plt.scatter(pred_tree,y_test)


# In[ ]:



knn = KNeighborsRegressor(n_neighbors=5)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=5')
print(r2_score(pred,y_test))
print(mean_absolute_error(pred,y_test))
print(mean_squared_error(pred,y_test))


# In[ ]:


plt.scatter(pred,y_test)


# In[ ]:




