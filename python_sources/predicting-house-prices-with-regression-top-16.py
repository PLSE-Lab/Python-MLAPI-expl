#!/usr/bin/env python
# coding: utf-8

# 

# With the dataset of housing prices, we will be:
# 
# 1. Loading our data.
# 2. Exploring the variables with visualization.
# 3. Cleaning the data.
# 4. Transforming the features.
# 5. Utilizing different regression models for our prediction.
# 

# In[ ]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# Save the ID columns for later and remove them from the datasets.

# In[ ]:


train_ID = train['Id']
test_ID = test['Id']

train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)


# Generate a correlation matrix for all variables

# In[ ]:


corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)


# We shall generate a scatter plot of the GrLivArea vs SalePrice.

# In[ ]:


fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# Remove the outliers and show the new scatterplot

# In[ ]:


train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# Plot of the distribution of the Sale Price and observe the skew and kurtosis

# In[ ]:


from scipy.stats import skew, kurtosis
sns.distplot(train['SalePrice']);
print(train["SalePrice"].skew(),"   ", train["SalePrice"].kurtosis())


# Take the log of the Sale Price in order to center the distribution

# In[ ]:


train["SalePrice"] = np.log1p(train["SalePrice"])
sns.distplot(train['SalePrice']);


# Check the skew and kurtosis after the log scale

# In[ ]:


print(train["SalePrice"].skew(),"  ", train["SalePrice"].kurtosis())


# Plot the variables most correlated to SalePrice Log in descending order

# In[ ]:


corr = train.corrwith(train['SalePrice']).abs().sort_values(ascending=False)[2:]

data = go.Bar(x=corr.index, 
              y=corr.values )
       
layout = go.Layout(title = 'Which variables are most correlated to Sale Price?', 
                   xaxis = dict(title = ''), 
                   yaxis = dict(title = 'correlation'),
                   autosize=False, width=750, height=500,)

fig = dict(data = [data], layout = layout)
iplot(fig)


# Concatenate both train and test values.

# In[ ]:


ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)


# Calculate the percentage of missing values in the dataset per feature.

# In[ ]:


all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data


# Plot the features by their missing data percentages in descending order

# In[ ]:


f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


# Now, we replace the missing values.

# In[ ]:


all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")


# Check if any missing values still exist

# In[ ]:


all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()


# Modify data types by converting them into strings

# In[ ]:


all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

all_data['OverallCond'] = all_data['OverallCond'].astype(str)

all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)


# Encode the labels in order to make categorical data more understandable.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))


# Get dummy variables and split the merged data into training and test sets

# In[ ]:


all_data = pd.get_dummies(all_data)

train = all_data[:ntrain]
test = all_data[ntrain:]


# Import libraries for modeling

# In[ ]:


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


# Create validation function with 5-fold validation

# In[ ]:


def rmsle(model):
    kfold = KFold(5, shuffle = True, random_state = 1).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring = "neg_mean_squared_error", cv = kfold))
    return(rmse)


# We will use the following regression models: Kernel Ridge, Lasso, ElasticNet, Gradient Boost, XGBoost, LighBGM, and Bayesian Ridge

# In[ ]:


KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree = 2, coef0 = 2.5)
KRR_score = rmsle(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(KRR_score.mean(), KRR_score.std()))


# In[ ]:


lasso = make_pipeline(RobustScaler(), Lasso(alpha = 0.0005, random_state = 1))
lasso_score = rmsle(lasso)
print("Lasso score: {:.4f} ({:.4f})\n".format(lasso_score.mean(), lasso_score.std()))


# In[ ]:


ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=1))
ENet_score = rmsle(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(ENet_score.mean(), ENet_score.std()))


# In[ ]:


GBoost = GradientBoostingRegressor(n_estimators = 3000, learning_rate = 0.05,
                                   max_depth = 4, max_features = 'sqrt',
                                   min_samples_leaf = 15, min_samples_split = 10, 
                                   loss = 'huber', random_state = 1)
GBoost_score = rmsle(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(GBoost_score.mean(), GBoost_score.std()))


# In[ ]:


model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
XGB_score = rmsle(model_xgb)
print("XGBoost score: {:.4f} ({:.4f})\n".format(XGB_score.mean(), XGB_score.std()))


# In[ ]:


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
lgb_score = rmsle(model_lgb)
print("LGB score: {:.4f} ({:.4f})\n".format(lgb_score.mean(), lgb_score.std()))


# In[ ]:


bayridge = BayesianRidge(compute_score=True)
bayridge_score = rmsle(bayridge)
print("Bayesian Ridge score: {:.4f} ({:.4f})\n".format(bayridge_score.mean(), bayridge_score.std()))


# Based on the errors, we will select all of the models, except for Kernel Ridge, and take their averages

# In[ ]:


LassoMd = lasso.fit(train.values,y_train)
ENetMd = ENet.fit(train.values,y_train)
GBoostMd = GBoost.fit(train.values,y_train)
XGBMd = model_xgb.fit(train.values,y_train)
LGBMd = model_lgb.fit(train.values,y_train)
BayRidgeMd = bayridge.fit(train.values,y_train)


# In[ ]:


final_model = (np.expm1(LassoMd.predict(test.values)) + np.expm1(ENetMd.predict(test.values)) + np.expm1(GBoostMd.predict(test.values)) + np.expm1(XGBMd.predict(test.values)) + np.expm1(LGBMd.predict(test.values)) + np.expm1(BayRidgeMd.predict(test.values)) ) / 6


# In[ ]:


submission = pd.DataFrame()
submission['Id'] = test_ID
submission['SalePrice'] = final_model
submission.to_csv('submission.csv',index=False)

