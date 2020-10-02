#!/usr/bin/env python
# coding: utf-8

# My first attempt at a regression model on Kaggle

# In[ ]:


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')

from scipy.stats import skew
from scipy.stats.stats import pearsonr

#sklearn
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# ## Import Data

# In[ ]:


train_in = pd.read_csv("../input/train.csv", header = 0, encoding = 'utf-8')


# In[ ]:


test_in = pd.read_csv("../input/test.csv", header = 0, encoding = 'utf-8')


# In[ ]:


train = train_in.copy()
test = test_in.copy()


# In[ ]:


pd.options.display.max_columns = 81


# In[ ]:


train.head()


# In[ ]:


train.info()


# A mix of int and object. With a few nulls that will need to be dealt with.

# ## Visualisations

# In[ ]:


plt.hist(train['SalePrice'])
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.show


# Pretty Skewed. Might be best to scale it later.

# ### Correlation Matrix

# In[ ]:


#correlation matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# We can see some intercorrelations between 'TotalBsmtSF' and '1stFlr', 'GarageCars' and 'GarageArea', 'YearBuilt' and 'GarageYrBlt. It will be best to remove some of these from the modelling. 
# 
# Focussing on the SalePrice column, we can see a few deep red boxes, indicating high correlation. 

# In[ ]:


corrmat.nlargest(12,'SalePrice')['SalePrice'].index


# In[ ]:


#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 13}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# This above plot shows the 10 features most correlated with SalePrice. As you can see some are correlated highly with themselves as well. We will drop those. Leaving us with 7 numerical features

# In[ ]:


#scatterplot
sns.set()
feat_cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[feat_cols], size = 2.5)
plt.show();


# In[ ]:


num_train = train[feat_cols]


# ## Data Cleaning

# ### Numerical Data

# In[ ]:


num_train.isnull().sum() # no nulls. good. 


# ### Null Data 

# In[ ]:



null_pc = train.isnull().sum()/len(train) 
null_pc[null_pc > 0.40]


# 4 features with greater than 40% nulls. Will drop those above 80%. FireplaceQu deserves further investigation

# In[ ]:


train['FireplaceQu'].value_counts()


# In[ ]:


train.groupby('FireplaceQu')['SalePrice'].mean().sort_values(ascending=False)


# No Suprises that the SalesPrice increases as the quality of Fireplace increases. Need to verify that the value is NA only when the number of fireplaces is 0. If this is the case can then replace the NAs with none and see where that fits in SalePrice-wise

# In[ ]:


train[train['Fireplaces'] == 0]['FireplaceQu'].isnull().sum()


# In[ ]:


train[train['FireplaceQu'].isnull()]['Fireplaces'].value_counts()


# In[ ]:


train['FireplaceQu'].fillna('None', inplace = True)


# In[ ]:


train.groupby('FireplaceQu')['SalePrice'].mean().sort_values(ascending=False)


# Interesting. Having no fireplace is better for your sale price than having a poor quality one. 

# In[ ]:


null_pc = train.isnull().sum()/len(train) 
null_list = null_pc[null_pc > 0.40].index.tolist()


# In[ ]:


train.drop(null_list, axis =1 , inplace = True)


# ### Object Data

# In[ ]:


#get dummies
obj_train = train.select_dtypes(include = ['object'])


# In[ ]:


obj_train.head()


# In[ ]:


obj_train = pd.get_dummies(obj_train)


# In[ ]:


obj_train.isnull().sum().sum()


# ### Modelling Prep

# In[ ]:


y_train = num_train.SalePrice
X_train = np.asmatrix(num_train.drop('SalePrice', axis = 1))


# In[ ]:


def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = 5))
    rmse = np.mean(rmse)
    return(rmse)


# ## Initial Modelling

# In[ ]:


lr = LinearRegression()


# In[ ]:


lr.fit(X_train, y_train)


# In[ ]:


rmse_cv(lr)


# In[ ]:


initial_pred = cross_val_predict(lr, X_train, y_train, cv = 5)


# In[ ]:


fig, ax = plt.subplots(figsize = (8,4))
ax.scatter(y_train, initial_pred)
ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


# Does relatively well early on, but fails to predict as the house price rises. Will look at using a ridge model before looking at unskewing data and then possible adding in the object data with dummies.

# ## Ridge Model

# In[ ]:


ridge_mod = Ridge()

ridge_mod.fit(X_train, y_train)
# In[ ]:


rmse_cv(ridge_mod)


# Barely a change, but a bit better.

# In[ ]:


ridge_pred = cross_val_predict(ridge_mod, X_train, y_train, cv = 5)


# In[ ]:


fig, ax = plt.subplots(figsize = (8,4))
ax.scatter(y_train, ridge_pred)
ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


# ## Dealing with Skewness

# In[ ]:


skewed_feats = num_train.apply(lambda x: skew(x.dropna())) #compute skewness


# In[ ]:


skewed_feats


# In[ ]:


skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index


# In[ ]:


skewed_feats


# In[ ]:


num_train[skewed_feats] = np.log1p(num_train[skewed_feats])


# In[ ]:


num_train.head()


# In[ ]:


fig, ax =plt.subplots(1, 2,figsize = (10,5))
plt.subplot(1, 2, 1)
plt.hist(num_train['SalePrice'])
plt.ylabel('Frequency')
plt.xlabel('Sale Price')
plt.subplot(1, 2, 2)
plt.hist(train['SalePrice']/1e5)
plt.xlabel('Sale Price [1e6]')
plt.ylabel('Frequency')
plt.show


# In[ ]:


y_train = num_train.SalePrice
X_train = np.asmatrix(num_train.drop('SalePrice', axis = 1))


# In[ ]:


sk_lr = LinearRegression()


# In[ ]:


sk_lr.fit(X_train, y_train)


# In[ ]:


rmse_cv(sk_lr)


# In[ ]:


skewed_pred = cross_val_predict(lr, X_train, y_train, cv = 5)


# In[ ]:


fig, ax = plt.subplots(figsize = (8,4))
ax.scatter(y_train, skewed_pred)
ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


# Dealing with the Skewness gives a much better model with basic Linear Regression. Now move onto Ridge.

# In[ ]:


sk_ridge_mod = Ridge()


# In[ ]:


sk_ridge_mod.fit(X_train, y_train)


# In[ ]:


rmse_cv(sk_ridge_mod) 


# In[ ]:


rmse_cv(sk_ridge_mod)  / rmse_cv(sk_lr)


# Almost exactly the same when dealing with skewness. Can play with the alpha.

# In[ ]:


alphas = [1e-6, 0.001, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]


# In[ ]:


cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation")
plt.xlabel("alpha")
plt.ylabel("rmse")


# Should look at adding in the object data to try and improve model further.

# ### Addition of object features

# In[ ]:


compl_train = pd.concat([num_train, obj_train], axis = 1)


# In[ ]:


num_train.shape, compl_train.shape


# In[ ]:


y_train = compl_train.SalePrice
X_train = compl_train.drop('SalePrice', axis = 1)


# In[ ]:


clr = LinearRegression()


# In[ ]:


clr.fit(X_train, y_train)


# In[ ]:


rmse_cv(clr)


# Pretty Poor, basic Linear Regression doesn't like the addition of all these features

# In[ ]:


c_ridge = Ridge()


# In[ ]:


c_ridge.fit(X_train, y_train)


# In[ ]:


rmse_cv(c_ridge)


# Now we are starting to see improvements.

# In[ ]:


c_pred = cross_val_predict(c_ridge, X_train, y_train, cv = 5)


# In[ ]:


fig, ax = plt.subplots(figsize = (8,4))
ax.scatter(y_train, c_pred)
ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


# In[ ]:


alphas = [1e-6, 0.001, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]


# In[ ]:


cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation")
plt.xlabel("alpha")
plt.ylabel("rmse")


# In[ ]:


cv_ridge.min()


# So an alpha of 0.5 would give us the best rmse from Ridge. Let's try other models

# ## Lasso Model
# Need to go back to all data as otherwise it gets messy predicting

# In[ ]:


all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))


# In[ ]:


numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])


# In[ ]:


all_data = pd.get_dummies(all_data)


# In[ ]:


all_data = all_data.fillna(all_data.mean())


# In[ ]:


X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice


# In[ ]:


lasso_model = LassoCV()


# In[ ]:


lasso_model.fit(X_train, y_train)


# In[ ]:


rmse_cv(lasso_model)


# We can improve this by tuning the alpha parameter again.

# In[ ]:


model_lasso = LassoCV(alphas = [10, 1, 0.1, 0.001, 0.0005]).fit(X_train, y_train)


# In[ ]:


rmse_cv(model_lasso).mean()


# In[ ]:


lasso_pred = cross_val_predict(model_lasso, X_train, y_train, cv = 5)


# In[ ]:


fig, ax = plt.subplots(figsize = (8,4))
ax.scatter(y_train, lasso_pred)
ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


# Looking good

# ## Final Predictions

# In[ ]:


lasso_preds = np.expm1(model_lasso.predict(X_test))


# In[ ]:


solution = pd.DataFrame({"id":test.Id, "SalePrice":lasso_preds})
solution.to_csv("housing_solution1.csv", index = False)


# ### Credit

# Thanks to Alexandru Papiu for his excellent kernel which was a great tutorial.
# 
# https://www.kaggle.com/apapiu/regularized-linear-models

# In[ ]:




