#!/usr/bin/env python
# coding: utf-8

# # Problem Statement
# 
# 
# A company is looking at prospective properties to buy to enter the market.
# 
# The company wants to know:
# 
# Which variables are significant in predicting the price of a house?
# 
# How well those variables describe the price of a house?

# ## Reading and Understanding the Data

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
import os

import re

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import scale
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

# hide warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


round(100*(train.isnull().sum())/len(train.index))


# In[ ]:


round(100*(test.isnull().sum())/len(test.index))


# In[ ]:


train.describe(include="all")


# In[ ]:


test.describe(include="all")


# # Data Cleaning & Preparation

# ### Outliers Treatment

# In[ ]:


fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# We can see at the bottom right two with extremely large GrLivArea that are of a low price. These values are huge oultliers. Therefore, we can safely delete them.

# In[ ]:


#Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# In[ ]:


#concatenate the train and test data in the same dataframe

ntrain = train.shape[0]
ntest = test.shape[0]
y = train.SalePrice.values
surprise = pd.concat((train, test)).reset_index(drop=True)
surprise.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(surprise.shape))


# In[ ]:


surprise_na = (surprise.isnull().sum() / len(surprise)) * 100
surprise_na = surprise_na.drop(surprise_na[surprise_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :surprise_na})
missing_data.head(20)


# In[ ]:


f, ax = plt.subplots(figsize=(10, 5))
plt.xticks(rotation='90')
sns.barplot(x=surprise_na.index, y=surprise_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


# In[ ]:


#Correlation map to see how features are correlated with SalePrice
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)


# ### Imputing missing values
# 
# 
# Example 
# 
# PoolQC : data description says NA means "No Pool"
# 
# Alley : data description says NA means - no alley access

# In[ ]:


for col in ('PoolQC','MiscFeature','Alley','Fence','FireplaceQu','MSSubClass','GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','BsmtQual', 'BsmtCond','MasVnrType','BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    surprise[col] = surprise[col].fillna('None')


# GarageYrBlt, GarageArea and GarageCars etc : Replacing missing data with 0 (Since No garage = no GarageYrBlt.)

# In[ ]:


for col in ('MasVnrArea', 'GarageYrBlt', 'GarageArea', 'GarageCars','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    surprise[col] = surprise[col].fillna(0)


# We will fill the missing value with most common ones.
# 
# Example 
# 
# MSZoning (The general zoning classification) : 'RL' is by far the most common value. So we can fill in missing values with 'RL'

# In[ ]:


surprise['MSZoning'] = surprise['MSZoning'].fillna(surprise['MSZoning'].mode()[0])
surprise['Electrical'] = surprise['Electrical'].fillna(surprise['Electrical'].mode()[0])
surprise['KitchenQual'] = surprise['KitchenQual'].fillna(surprise['KitchenQual'].mode()[0])
surprise['Exterior1st'] = surprise['Exterior1st'].fillna(surprise['Exterior1st'].mode()[0])
surprise['Exterior2nd'] = surprise['Exterior2nd'].fillna(surprise['Exterior2nd'].mode()[0])
surprise['SaleType'] = surprise['SaleType'].fillna(surprise['SaleType'].mode()[0])


# Utilities : For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling. We can then safely remove it.

# In[ ]:


surprise = surprise.drop(['Utilities'], axis=1)


# In[ ]:


#Functional : data description says NA means typical

surprise["Functional"] = surprise["Functional"].fillna("Typ")


# LotFrontage : Since the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood , we can fill in missing values by the median LotFrontage of the neighborhood.

# In[ ]:



surprise["LotFrontage"] = surprise.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))


# In[ ]:


surprise.isnull().values.any()


# ### Dealing with Categorical Fields
# 
# #### Let's now prepare the data and build the model.
# 

# In[ ]:


#Transforming some numerical variables that are really categorical
surprise['MSSubClass'] = surprise['MSSubClass'].apply(str)
surprise['OverallCond'] = surprise['OverallCond'].astype(str)
surprise['YrSold'] = surprise['YrSold'].astype(str)
surprise['MoSold'] = surprise['MoSold'].astype(str)


# In[ ]:


surprise.shape


# In[ ]:


surprise = pd.get_dummies(surprise)
print(surprise.shape)


# In[ ]:


train = surprise[:ntrain]
test = surprise[ntrain:]


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


# scaling the features
from sklearn.preprocessing import scale

# storing column names in cols, since column names are (annoyingly) lost after 
# scaling (the df is converted to a numpy array)
cols = train.columns
train = pd.DataFrame(scale(train))
train.columns = cols
train.columns


# In[ ]:


# split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, y, 
                                                    train_size=0.7,
                                                    test_size = 0.3, random_state=100)


# ##  Model Building and Evaluation

# ## Logistic Regression, Ridge and Lasso Regression

# In[ ]:


# list of alphas to tune
params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500,1000 ]}


ridge = Ridge()

# cross validation
folds = 5
model_cv = GridSearchCV(estimator = ridge, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            
model_cv.fit(X_train, y_train) 


# In[ ]:


cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results = cv_results[cv_results['param_alpha']<=200]
cv_results.head()


# In[ ]:


# plotting mean test and train scoes with alpha 
cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')

# plotting
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')
plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper left')
plt.show()


# In[ ]:


model_cv.best_params_


# In[ ]:


alpha = 500
ridge = Ridge(alpha=alpha)

ridge.fit(X_train, y_train)


# In[ ]:


ridge.coef_ 


# ## Lasso

# In[ ]:


# lasso regression
lm = Lasso(alpha=1000)
lm.fit(X_train, y_train)

# predict
y_train_pred = lm.predict(X_train)
print(r2_score(y_true=y_train, y_pred=y_train_pred))
y_test_pred = lm.predict(X_test)
print(r2_score(y_true=y_test, y_pred=y_test_pred))


# In[ ]:


# lasso model parameters
model_parameters = list(lm.coef_)
model_parameters.insert(0, lm.intercept_)
model_parameters = [round(x, 3) for x in model_parameters]
cols = train.columns
cols = cols.insert(0, "constant")
list(zip(cols, model_parameters))


# In[ ]:


lasso = Lasso()

# cross validation
model_cv = GridSearchCV(estimator = lasso, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            

model_cv.fit(X_train, y_train) 


# In[ ]:


cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results.head()


# In[ ]:


# plot
cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('r2 score')
plt.xscale('log')
plt.show()


# In[ ]:


# plotting mean test and train scoes with alpha 
cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')

# plotting
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')

plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper left')
plt.show()


# In[ ]:


model_cv.best_params_


# In[ ]:


alpha = 1000

lasso = Lasso(alpha=alpha)
        
lasso.fit(X_train, y_train)


# In[ ]:


lasso.coef_


# ### The company wants to know:
# 
# Which variables are significant in predicting the price of a house?
# 
# How well those variables describe the price of a house?
# 
# #### For this we'll see the Positive coefficients
# 

# In[ ]:


# lasso model parameters
model_parameters = list(lm.coef_)
model_parameters.insert(0, lm.intercept_)
model_parameters = [round(x, 3) for x in model_parameters]
cols = train.columns
cols = cols.insert(0, "constant")
list(zip(cols, model_parameters))


# In[ ]:


# lasso regression
lm1 = Lasso(alpha=1000)
lm1.fit(X_train, y_train)

# predict
y_train_pred = lm1.predict(X_train)
print(r2_score(y_true=y_train, y_pred=y_train_pred))
y_test_pred = lm1.predict(X_test)
print(r2_score(y_true=y_test, y_pred=y_test_pred))


# In[ ]:


# Ridge regression
lm2 = Ridge(alpha=500)
lm2.fit(X_train, y_train)

# predict
y_train_pred = lm2.predict(X_train)
print(r2_score(y_true=y_train, y_pred=y_train_pred))
y_test_pred = lm2.predict(X_test)
print(r2_score(y_true=y_test, y_pred=y_test_pred))


# In[ ]:


preds = lm1.predict(test)
sub = pd.DataFrame()
sub['Id'] = test['Id']
sub['SalePrice'] = preds
sub.to_csv('house_sub.csv',index=False)

