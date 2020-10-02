#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


df = pd.read_csv('../input/insurance.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df['sex'].value_counts()


# In[ ]:


df['smoker'].value_counts()


# In[ ]:


df[df.duplicated()]


# In[ ]:


df.drop_duplicates(inplace=True)


# In[ ]:


cat_cols = df.select_dtypes(exclude = 'number')


# In[ ]:


num_cols = df.select_dtypes(include = 'number')


# In[ ]:


onehot_cat_cols = pd.get_dummies(cat_cols)


# In[ ]:


onehot_cat_cols.head()


# In[ ]:


df_final = pd.concat([num_cols,onehot_cat_cols],sort=True,axis=1)


# In[ ]:


df_final.head()


# In[ ]:


X = df_final.drop('expenses',axis=1)


# In[ ]:


y = df_final['expenses']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train , X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)


# In[ ]:


X_train.shape


# In[ ]:


y_train.shape


# In[ ]:


X_test.shape


# In[ ]:


y_test.shape


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor


# In[ ]:


from sklearn.metrics import mean_squared_error 
from math import sqrt,ceil
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


l = ceil(sqrt(df.shape[0]))


# In[ ]:


rmse = []
for k in range(0,l+1):
    k = k+1
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train,y_train)
    y_test_pred = model.predict(X_test)
    rmse_error = sqrt(mean_squared_error(y_test,y_test_pred))
    rmse.append(rmse_error)
    print('RMSE value for k=' , k , 'is:', rmse_error)


# In[ ]:


min(rmse)


# In[ ]:


# from sklearn.model_selection import GridSearchCV
# params = {'n_neighbors':[2,3,4,5,6,7,8,9]}

# knn = KNeighborsRegressor()

# model = GridSearchCV(knn, params, cv=5)
# model.fit(X_train,y_train)
# model.best_params_


# In[ ]:


error_curve = pd.DataFrame(rmse,columns=['error'])


# In[ ]:


error_curve.head()


# In[ ]:


error_curve.plot()


# In[ ]:


RMSE is minimum at k = 4

