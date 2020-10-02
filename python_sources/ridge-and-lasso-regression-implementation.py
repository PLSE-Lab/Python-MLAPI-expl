#!/usr/bin/env python
# coding: utf-8

# # Ridge and Lasso Regression Implementation

# In[ ]:


from sklearn.datasets import load_boston


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib as plt 


# In[ ]:


df = load_boston()
df


# In[ ]:


dataset = pd.DataFrame(df.data)  # Independent features
dataset.head()


# In[ ]:


dataset.columns = df.feature_names
dataset.head()


# In[ ]:


df.target.shape


# In[ ]:


dataset['Price'] = df.target
dataset.head()


# In[ ]:


X = dataset.iloc[:,:-1] # independent features
y = dataset.iloc[:,-1] # dependent features


# ### Linear Regression

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

linear_regressor = LinearRegression()
mse = cross_val_score(linear_regressor, X, y, scoring='neg_mean_squared_error', cv=5)
mean_mse = np.mean(mse)
print(mean_mse)


# ### Ridge Regression

# In[ ]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge = Ridge()
parameters = {'alpha':[1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]}
ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
ridge_regressor.fit(X, y)

print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


# ### Lasso Regression

# In[ ]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

lasso = Lasso()
parameters = {'alpha':[1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]}
lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)
lasso_regressor.fit(X, y)

print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[ ]:


prediction_lasso = lasso_regressor.predict(X_test)
prediction_ridge = ridge_regressor.predict(X_test)


# In[ ]:


import seaborn as sns

sns.distplot(y_test-prediction_lasso)


# In[ ]:


sns.distplot(y_test-prediction_ridge)


# In[ ]:




