#!/usr/bin/env python
# coding: utf-8

# The goal of this kernel is to get a model that provides a reasonable prediction score without applying a thorough model implementation. 
# 
# I employed a quick EDA, so that the feature variables used were determined from a quick glance. I also restricted the number of model libraries from scikit-learn. The best score obtained with this pragmatic approach was 0.87 on the test set by using Gradient Boosted Regression Trees.   
# 
# The feature variables selected, allowed in particular for no pretreatment such as normalization or standardization. Decision trees work well when there are variables with totally different scales.
# 
# Your comments and votes make us all grow ! Cheers ! 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df_train.head(3)


# In[ ]:


df_train.info()


# In[ ]:


# Determine correlation and derive feature variables from it.
# Variables providing less than 0.50 to be discarded. 
df_train.corr()[['SalePrice']].sort_values(by='SalePrice', ascending=False).head(11)


# In[ ]:


# Fill missing values for the variables of interest:
df_train['GarageYrBlt'].fillna(round(df_train['GarageYrBlt'].median(), 1), inplace=True)
df_train['MasVnrArea'].fillna(0.0, inplace=True)


# In[ ]:


# A simple regression plot to visualize correlations on each feature variable to be used:
def regplot(x):
    sns.regplot(x, y=df_train['SalePrice'], data=df_train)


# In[ ]:


regplot(x='OverallQual')
plt.ylim(0, )


# In[ ]:


regplot(x='GrLivArea')
plt.ylim(0, )


# In[ ]:


regplot(x= 'GarageCars')
plt.ylim(0, )


# In[ ]:


regplot(x='GarageArea')
plt.ylim(0, )


# In[ ]:


regplot(x= 'TotalBsmtSF')
plt.ylim(0, )


# In[ ]:


regplot(x='1stFlrSF')
plt.ylim(0, )


# In[ ]:


regplot(x='FullBath')
plt.ylim(0, )


# In[ ]:


regplot(x= 'TotRmsAbvGrd')
plt.ylim(0, )


# In[ ]:


regplot(x = 'YearBuilt')
plt.ylim(0, )


# In[ ]:


regplot(x= 'YearRemodAdd')
plt.ylim(0, )


# In[ ]:


X = df_train[['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath',
         'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MasVnrArea', 'Fireplaces', 'BsmtFinSF1']]
# verify for null values
X.info()


# In[ ]:


# define target variable
y = df_train[['SalePrice']]
y.info()


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# In[ ]:


lr = LinearRegression().fit(X_train, y_train)


# In[ ]:


print('lr.coef_: {}'.format(lr.coef_))
print('lr.intercept_: {}'.format(lr.intercept_))


# In[ ]:


print('training set score: {}'.format(lr.score(X_train, y_train)))
print('test set score: {}'.format(lr.score(X_test, y_test)))


# In[ ]:


from sklearn.linear_model import Ridge
ridge = Ridge(alpha=.1)
ridge.fit(X_train, y_train)
print('training set score: {}'.format(ridge.score(X_train, y_train)))
print('test set score: {}'.format(ridge.score(X_test, y_test)))


# In[ ]:


from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1, max_iter=100000)
lasso.fit(X_train, y_train)
print('training set score: {}'.format(lasso.score(X_train, y_train)))
print('test set score: {}'.format(lasso.score(X_test, y_test)))
print('number of features used: {}'.format(np.sum(lasso.coef_ != 0)))


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor()
tree.fit(X_train, y_train)
print('training set score: {}'.format(tree.score(X_train, y_train)))
print('test set score: {}'.format(tree.score(X_test, y_test)))


# As seen above, one inconvenience of decision trees they tend to overfit our training set. To respond to this problem
# lets use random forest.

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor(random_state=0)
gbrt.fit(X_train, y_train)
print('training set score: {}'.format(gbrt.score(X_train, y_train)))
print('test set score: {}'.format(gbrt.score(X_test, y_test)))


# In[ ]:


# this plot / algorithm referenced from the book: Le Machine Learning avec Python / O'Reilly / 2018

def plot_feature_importance(model):
    n_features = X.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X.columns)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.ylim(-1, n_features)
plot_feature_importance(gbrt)

