#!/usr/bin/env python
# coding: utf-8

# ## References 
# These were some of the notebooks that really helped me with mine, definitely check them out for a more detailed experience.
# 
# https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
# 
# https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard

# **Import necessary libraries**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()
import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
pd.set_option('display.max_colwidth', -1) #to max the column width


# Import our Data

# In[ ]:


train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
train.head(4)


# In[ ]:


train.columns


# This is a regression problem, what we're trying to predict is the SalesPrice, so lets take a look at it!

# In[ ]:


train['SalePrice'].describe()


# In[ ]:


plt.figure(figsize=(16, 10))
plt.title('Sales Price Distribution')
sns.distplot(train['SalePrice']);


# From our distribution plot above, we can see that we have a peaked distribution that is not normal with positive skewness!
# 
# Therefore, before performing regression, our data has to be transformed, a logrithimic transformation should suffice.

# In[ ]:


target = np.log1p(train.SalePrice)

plt.figure(figsize=(16, 10))
plt.title('Transformed Sales Price Distribution')
sns.distplot(target);


# Lets examine our numerical features!

# In[ ]:


num_feats = train.select_dtypes(include=[np.number])
print(num_feats.shape)
num_feats.columns


# Now for the categorical features

# In[ ]:


cate_feats = train.select_dtypes(include=[np.object])
print(cate_feats.shape)
cate_feats.columns


# # Data Exploration

# **Correlation with Numerical features and SalesPrice**

# In[ ]:


num_corr = num_feats.corr()
print(num_corr['SalePrice'].sort_values(ascending = False),'\n')


# Lets better visualize the correlation between the more highly correlated features and **SalePrice**

# In[ ]:


k= 11 #all correlations that are 0.5 and above
cols = num_corr.nlargest(k,'SalePrice')['SalePrice'].index
print(cols)
data = np.corrcoef(train[cols].values.T)
plt.figure(figsize=(14, 7))
plt.title('Correlation of Numerical Features')
sns.heatmap(data, vmax=.8, linewidths=0.01,square=True,annot=True,cmap='viridis',
            xticklabels=cols.values ,annot_kws = {'size':12},yticklabels = cols.values)


# From the above heatmap, we can see that some other features are highly correlated with one another. 
# 
# 1. **TotalBsmtSF** and **1stFlrSF**, 
# 2. **GarageArea** and **GarageCars**, 
# 3. **TotRmsAbvGrd** and **GrLivArea**

# **SCATTERPLOT**
# 
# A scatterplot can help us better visualise the correlation between SalePrice and its most correlated features and help us identify outliers if any!

# Due to the high correlation observed between some features, we'll be using those that have a higher correlation with **SalePrice**. 
# For example, we'll use **GrLivArea** and not **TotRmsAbvGrd** since they're both highly correlated with each but the former is better correlated with **SalePrice**

# In[ ]:


fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6))= plt.subplots(nrows=3,ncols=2,figsize=(15, 10))

sns.regplot(x='OverallQual', y='SalePrice', data=train, ax=ax1)

sns.regplot(x='GrLivArea', y='SalePrice', data=train, ax=ax2)

sns.regplot(x='GarageCars', y='SalePrice', data=train, ax=ax3)

sns.regplot(x='TotalBsmtSF', y='SalePrice', data=train, ax=ax4)

sns.regplot(x='FullBath', y='SalePrice', data=train, ax=ax5)

sns.regplot(x='YearRemodAdd', y='SalePrice', data=train, ax=ax6)

plt.show()

sns.regplot(x='YearBuilt', y='SalePrice', data=train)


# It would seem as though our features exhibit a linear relationship with the **SalePrice** and we can observe outliers in **GrLivArea** and **TotalBsmtSF** scatter plots, we'll deal with those later.

# **Categorical features**

# First, lets take a look at how many null values are in our categorical features

# In[ ]:


cate_null = cate_feats.isnull().sum().sort_values(ascending=False)
null_percent = (cate_feats.isnull().sum()/cate_feats.isnull().count() * 100).sort_values(ascending=False)
missing_data = pd.concat([cate_null, null_percent], axis=1, keys=['Total Null Values', 'Percentage'])

missing_data.head(20)


# From the decription, we know that if some features have "nan" values, it means that the said feature doesn't exist for the corresponding observation. Therfore, we can replace said "nan" values with "None".
# But for the features with more than 10% missing values, we'll probably just drop them, they most likely wont impact our model in anyway.

# In[ ]:


dropped = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
cate_feats.drop(labels=dropped, axis=1, inplace=True)
cate_feats.shape


# In[ ]:


# Since Electrical has just one null value...
cate_feats.Electrical.value_counts()


# In[ ]:


train.GarageCond = cate_feats['GarageCond'].fillna(value='None', inplace=True)
train.GarageQual = cate_feats['GarageQual'].fillna(value='None', inplace=True)
train.GarageFinish = cate_feats['GarageFinish'].fillna(value='None', inplace=True)
train.GarageType = cate_feats['GarageType'].fillna(value='None', inplace=True)
train.BsmtFinType2 = cate_feats['BsmtFinType2'].fillna(value='None', inplace=True)
train.BsmtExposure = cate_feats['BsmtExposure'].fillna(value='None', inplace=True)
train.BsmtQual = cate_feats['BsmtQual'].fillna(value='None', inplace=True)
train.BsmtFinType1 = cate_feats['BsmtFinType1'].fillna(value='None', inplace=True)
train.BsmtCond = cate_feats['BsmtCond'].fillna(value='None', inplace=True)
train.MasVnrType = cate_feats['MasVnrType'].fillna(value='None', inplace=True)
train.Electrical = cate_feats['Electrical'].fillna(value='SBrkr', inplace=True)


# Lets see if all our categorical features have unique values

# In[ ]:


unique_list=[]
for column in cate_feats.columns:
    no_of_unique = len(cate_feats[column].unique())
    unique_list.append(no_of_unique)
    print("The '{a}' column has {b} unique values".format(a=column, b=no_of_unique))
print("Total number of categories with unique values is {c}".format(c=len(unique_list)))


# Looking at the categories, the most likely to have an effect on price would be the neighborhood in which they are situated
# Now, we'll visualize the correlation between **Neighborhood** and our **SalePrice**

# In[ ]:


plt.figure(figsize=(16, 10))
plt.xticks(rotation=90)
sns.boxplot(x='Neighborhood', y='SalePrice', data=train)


# # Feature Selection

# After exploring our training data extensively, we can finally select the features deemed most important for our final training feature matrix.

# In[ ]:


bbb = ['Id','OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearRemodAdd', 'YearBuilt', 'SalePrice']
num_feats = num_feats[bbb]
num_feats.head()


# In[ ]:


train_data = pd.concat([num_feats, cate_feats], axis=1)
train_data.shape


# **Removing Outliers**
# 
# From our data exploration, we remember that the features  **GrLivArea** and **TotalBsmtSF** had outliers, lets sort those out!

# In[ ]:


fig, ((ax1, ax2)) = plt.subplots(nrows=2,ncols=1,figsize=(11, 7))

sns.regplot(x='TotalBsmtSF', y='SalePrice', data=train_data, ax=ax1)

sns.regplot(x='GrLivArea', y='SalePrice', data=train_data, ax=ax2)


# In[ ]:


train_data = train_data.drop(train_data[train_data['GrLivArea'] > 4500].index)
train_data = train_data.drop(train_data[train_data['TotalBsmtSF'] > 6000].index)
train_data.head()


# **Test Data**

# In[ ]:


cols = list(train_data.columns)
print(cols[8])
cols.pop(8)
cols[:10]


# In[ ]:


test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv', usecols=cols)[cols]
print(test_data.shape)
test_data.head()


# In[ ]:


null_test = test_data.isnull().sum().sort_values(ascending=False)
percentage = (test_data.isnull().sum()/test_data.isnull().count() * 100).sort_values(ascending=False)
missing_data = pd.concat([null_test, percentage], axis=1, keys=['Total Null Values', 'Percentage'])

missing_data.head(20)


# In[ ]:


#Replacing the higher missing values with "None" as specified by the documentation.
test_data.GarageCond.fillna(value='None', inplace=True)
test_data.GarageQual.fillna(value='None', inplace=True)
test_data.GarageFinish.fillna(value='None', inplace=True)
test_data.GarageType.fillna(value='None', inplace=True)
test_data.BsmtFinType2.fillna(value='None', inplace=True)
test_data.BsmtExposure.fillna(value='None', inplace=True)
test_data.BsmtQual.fillna(value='None', inplace=True)
test_data.BsmtFinType1.fillna(value='None', inplace=True)
test_data.BsmtCond.fillna(value='None', inplace=True)
test_data.MasVnrType.fillna(value='None', inplace=True)

#Replacing the low missing values with the most common.
test_data.Exterior1st.fillna(value=test_data.Exterior1st.value_counts().idxmax(), inplace=True)
test_data.TotalBsmtSF.fillna(value=test_data.TotalBsmtSF.median(), inplace=True)
test_data.Exterior2nd.fillna(value=test_data.Exterior2nd.value_counts().idxmax(), inplace=True)
test_data.SaleType.fillna(value=test_data.SaleType.value_counts().idxmax(), inplace=True)
test_data.KitchenQual.fillna(value=test_data.KitchenQual.value_counts().idxmax(), inplace=True)
test_data.GarageCars.fillna(value=test_data.GarageCars.median(), inplace=True)
test_data.Utilities.fillna(value=test_data.Utilities.value_counts().idxmax(), inplace=True)
test_data.Functional.fillna(value=test_data.Functional.value_counts().idxmax(), inplace=True)
test_data.MSZoning.fillna(value=test_data.MSZoning.value_counts().idxmax(), inplace=True)


# **Encoding Categorical Variables**

# In[ ]:


a = train_data.select_dtypes(include=[np.object])
cols_list = list(a.columns)
train = train_data
test = test_data


# In[ ]:


from sklearn.preprocessing import LabelEncoder
train[cols_list] = train[cols_list].apply(LabelEncoder().fit_transform)
test[cols_list] = test[cols_list].apply(LabelEncoder().fit_transform)

train.head()


# In[ ]:


test.head()


# # Model Selection and Training

# Lets import our tools for training and predictions

# In[ ]:


from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin


# **Train-test split**

# In[ ]:


X = train.drop(['Id', 'SalePrice'], axis=1)
y = np.log1p(train.SalePrice)
z = test.drop(['Id'], axis=1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# We'll scale the data using the **RobustScaler** module or **StandardScaler** to improve predictions basically!

# In[ ]:


scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#scale the prediction data
pred_test = scaler.transform(z)


# **Linear Regression**

# In[ ]:


Linear_reg = LinearRegression()
Linear_reg.fit(X_train, y_train)
Linear_pred = Linear_reg.predict(X_test)
Linear_mse = np.sqrt(mean_squared_error(y_test, Linear_pred))
Linear_r2 = r2_score(y_test, Linear_pred)

print('The Linear model has a root mean squared error of: {}'.format(Linear_mse))
print('The Linear model has an r2 score of: {}'.format(Linear_r2))


# In[ ]:


kfolds = KFold(n_splits=10, shuffle=True, random_state=42)


# **Ridge**

# In[ ]:


ridge = Ridge()
param_ridge = {'alpha': [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]}
ridge_model = GridSearchCV(ridge, param_ridge, cv=kfolds, n_jobs=2)
ridge_model.fit(X_train, y_train)
print(ridge_model.best_params_)


# In[ ]:


ridge_model = ridge_model.best_estimator_
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)
ridge_mse = np.sqrt(mean_squared_error(y_test, ridge_pred))
ridge_r2 = r2_score(y_test, ridge_pred)

print('The Ridge model has a root mean squared error of: {}'.format(ridge_mse))
print('The Ridge model has an r2 score of: {}'.format(ridge_r2))


# **Lasso**

# In[ ]:


lasso = Lasso(max_iter=1e7, random_state=42)
param_lasso = {'alpha': [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]}
lasso_model = GridSearchCV(lasso, param_lasso, cv=kfolds, n_jobs=2)
lasso_model.fit(X_train, y_train)
print(lasso_model.best_params_)


# In[ ]:


lasso_model = lasso_model.best_estimator_
lasso_model.fit(X_train, y_train)
lasso_pred = lasso_model.predict(X_test)
lasso_mse = np.sqrt(mean_squared_error(y_test, lasso_pred))
lasso_r2 = r2_score(y_test, lasso_pred)

print('The Lasso model has a root mean squared error of: {}'.format(lasso_mse))
print('The Lasso model has an r2 score of: {}'.format(lasso_r2))


# **SVR**

# In[ ]:


svr_model = SVR(C= 20, epsilon= 0.001, gamma=0.0009)
svr_model.fit(X_train, y_train)
svr_pred = svr_model.predict(X_test)
svr_mse = np.sqrt(mean_squared_error(y_test, svr_pred))
svr_r2 = r2_score(y_test, svr_pred)

print('The SVR model has a root mean squared error of: {}'.format(svr_mse))
print('The SVR model has an r2 score of: {}'.format(svr_r2))


# **ElasticNet**

# In[ ]:


elastic_net = ElasticNet(max_iter=1e7, random_state=42)
param_enet = {'alpha': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007],
              'l1_ratio': [0.8, 0.85, 0.9, 0.95, 0.99, 1]}
enet_model = GridSearchCV(elastic_net, param_enet, cv=5, n_jobs=2)
enet_model.fit(X_train, y_train)
print(enet_model.best_params_)


# In[ ]:


enet_model = enet_model.best_estimator_
enet_model.fit(X_train, y_train)
enet_pred = enet_model.predict(X_test)
enet_mse = np.sqrt(mean_squared_error(y_test, enet_pred))
enet_r2 = r2_score(y_test, enet_pred)

print('The ElasticNet model has a root mean squared error of: {}'.format(enet_mse))
print('The ElasticNet model has an r2 score of: {}'.format(enet_r2))


# **Gradient Boosting Regressor**

# In[ ]:


gboost_model = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, 
                                         max_features='sqrt', min_samples_leaf=15, min_samples_split=10, 
                                         loss='huber', random_state =42)
gboost_model.fit(X_train, y_train)
gboost_pred = gboost_model.predict(X_test)
gboost_mse = np.sqrt(mean_squared_error(y_test, gboost_pred))
gboost_r2 = r2_score(y_test, gboost_pred)

print('The Gradient Boosting Regressor has a root mean squared error of: {}'.format(gboost_mse))
print('The Gradient Boosting Regressor has an r2 score of: {}'.format(gboost_r2))


# # Stacking Models

# In[ ]:


class StackModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    def fit(self, X, y):
        self.models_ = [x for x in self.models]
        
        for model in self.models_:
            model.fit(X, y)

        return self
    
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)


# In[ ]:


stack_model = StackModels(models= (gboost_model, svr_model, enet_model, lasso_model))

stack_model.fit(X_train, y_train)
stack_pred = stack_model.predict(X_test)
stack_mse = np.sqrt(mean_squared_error(y_test, stack_pred))
stack_r2 = r2_score(y_test, stack_pred)

print('The Stacked Model has a root mean squared error of: {}'.format(stack_mse))
print('The Stacked Model has an r2 score of: {}'.format(stack_r2))


# # Predictions and Submission

# In[ ]:


target = stack_model.predict(pred_test)
predictions = np.exp(target)

submission = pd.DataFrame({'Id': test['Id'],
                          'SalePrice': predictions})


# In[ ]:


submission.to_csv('Submission_stack.csv', index=False)
submission.head()


# In[ ]:


get_ipython().run_line_magic('pinfo', 'SVR')


# In[ ]:




