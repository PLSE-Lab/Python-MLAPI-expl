#!/usr/bin/env python
# coding: utf-8

# **Predict Housing Prices (so you can get a better  house)**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #data science plot

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit

import matplotlib.pyplot as plt
# Any results you write to the current directory are saved as output.

#Ref - https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python


# Load train and test data frames and combine them to a single data frame.

# In[ ]:


dataFrame_train = pd.read_csv('../input/train.csv')
dataFrame_test = pd.read_csv('../input/test.csv')
dataFrame = pd.concat([dataFrame_train, dataFrame_test])


# In[ ]:


dataFrame.head(5)


# Load sale price into y vector, and drop it from the dataFrame

# In[ ]:


y = dataFrame_train['SalePrice']
dataFrame = dataFrame.drop(['SalePrice', 'Utilities', 'Id'], axis=1)


# Look at how much data we got, and Identify columns with missing data.

# In[ ]:


dataFrame.info()


# In[ ]:


dataFrame.columns[dataFrame.isna().any()].tolist()


# Replace missing values with NA values when NA means something, esle replace it with mean/most frequent values.

# In[ ]:


dataFrame.Alley.fillna('NA', inplace=True)
dataFrame.BsmtCond.fillna('NA', inplace=True)
dataFrame.BsmtExposure.fillna('NA', inplace=True)
dataFrame.BsmtFinSF1.fillna(0, inplace=True)
dataFrame.BsmtFinSF2.fillna(0, inplace=True)
dataFrame.BsmtFinType1.fillna('NA', inplace=True)
dataFrame.BsmtFinType2.fillna('NA', inplace=True)
dataFrame.BsmtFullBath.fillna(0, inplace=True)
dataFrame.BsmtHalfBath.fillna(0, inplace=True)
dataFrame.BsmtQual.fillna("NA", inplace=True)
dataFrame.BsmtUnfSF.fillna(0, inplace=True)
dataFrame.Electrical.fillna('SBrkr', inplace=True)
dataFrame.Exterior1st.fillna('VinylSd', inplace=True)
dataFrame.Exterior2nd.fillna('VinylSd', inplace=True)
dataFrame.Fence.fillna('NA', inplace=True)
dataFrame.FireplaceQu.fillna('NA', inplace=True)
dataFrame.Functional.fillna('Typ', inplace=True)
dataFrame.GarageArea.fillna(0, inplace=True)
dataFrame.GarageCond.fillna('NA', inplace=True)
dataFrame.GarageCars.fillna(0, inplace=True)
dataFrame.GarageYrBlt.fillna(0, inplace=True)
dataFrame.GarageQual.fillna("NA", inplace=True)
dataFrame.GarageFinish.fillna("NA", inplace=True)
dataFrame.GarageType.fillna("NA", inplace=True)
dataFrame.KitchenQual.fillna("TA", inplace=True)
dataFrame.LotFrontage.fillna(dataFrame.LotFrontage.mean(), inplace=True)
dataFrame.MSZoning.fillna("RL", inplace=True)
dataFrame.MasVnrType.fillna("None", inplace=True)
dataFrame.MasVnrArea.fillna(0, inplace=True)
dataFrame.MiscFeature.fillna('NA', inplace=True)
dataFrame.PoolQC.fillna('NA', inplace=True)
dataFrame.SaleType.fillna("WD", inplace=True)
dataFrame.TotalBsmtSF.fillna(0, inplace=True)
#dataFrame.Utilities.fillna('AllPub', inplace=True)


# Calculate a new feature `houseAge` which determines an average age of the house since its been built/remodeled.

# In[ ]:


#dataFrame['houseAge'] = dataFrame['YrSold'] -( dataFrame['YearRemodAdd'] + dataFrame['YearBuilt'])/2.0


# Convert all categorical variables to dummy variables.

# In[ ]:


#Columns which have integer values but are actually categorical
int_to_cat_columns = ['MSSubClass']


# In[ ]:


cat_columns = dataFrame.select_dtypes(include='object').columns.values.tolist()
dataFrame = pd.get_dummies(dataFrame, columns=cat_columns+int_to_cat_columns)


# In[ ]:


dataFrame.columns.tolist()


# Create a correlation matrix and generate a heatmap

# In[ ]:





# In[ ]:


#Rebuild train and test data frames
dataFrame_trainval = dataFrame.iloc[:y.shape[0], :]
X_trainval = dataFrame_trainval.values
dataFrame_trainval['SalePrice'] = y
dataFrame_test = dataFrame.iloc[y.shape[0]:, :]
X_test = dataFrame_test.values


# corr = dataFrame_train.corr()
# 
# # Set up the matplotlib figure
# #f, ax = plt.subplots(figsize=(30, 20))
# 
# # Generate a custom diverging colormap
# #cmap = sns.diverging_palette(359, -359, as_cmap=True)
# 
# # Draw the heatmap with the mask and correct aspect ratio
# #sns.heatmap(corr, cmap=cmap, vmax=1, vmin=-1, center=0,
# #            square=True, linewidths=1)

# Build different model versions with different features, chosen based on correlation greater than a specific value.****

# threshold_versions = [0.5, 0.4, 0.3, 0.2, 0.1]
# salePrice_correlations = corr['SalePrice'] 
# feature_versions = []
# for threshold in threshold_versions:
#     features = dataFrame_train.columns[(salePrice_correlations > threshold) & (salePrice_correlations < 1)]
#     feature_versions.append(features)

# Run Linear Regression and Gradient Boosting models on each of the feature versions

# For each set of features, build training and test matrices X and y, run the models, create a new data Frame with the predictions and save it to a csv.

# for idx, features in enumerate(feature_versions):
#     lr = LinearRegression()
#     X_train = dataFrame_train.loc[:, features]
#     y_train = dataFrame_train['SalePrice']
#     X_test = dataFrame_test.loc[:, features]
#     lr.fit(X_train, y_train)
#     y_pred = lr.predict(X_test)
#     res = pd.DataFrame()
#     res['Id'] = dataFrame_test['Id']
#     res['SalePrice'] = y_pred
#     res.to_csv('lr_{}.csv'.format(idx), index=False)

# for idx, features in enumerate(feature_versions):
#     xgb = XGBRegressor()
#     X_train = dataFrame_train.loc[:, features]
#     y_train = dataFrame_train['SalePrice']
#     X_test = dataFrame_test.loc[:, features]
#     xgb.fit(X_train, y_train)
#     y_pred = xgb.predict(X_test)
#     res = pd.DataFrame()
#     res['Id'] = dataFrame_test['Id']
#     res['SalePrice'] = y_pred
#     res.to_csv('xgb_{}.csv'.format(idx), index=False)

# rfr = RandomForestRegressor(n_jobs=-1)

# rfr.fit(dataFrame_train.drop('SalePrice', axis=1), y)

# y_pred = rfr.predict(dataFrame_test)

# In[ ]:


splitter = ShuffleSplit(n_splits=1, test_size=0.2)
splits = list(splitter.split(X_trainval, y))[0]
train_ind, test_ind = splits


# In[ ]:


X_train = X_trainval[train_ind]
X_val  = X_trainval[test_ind]

y_train = y[train_ind]
y_val  = y[test_ind]


# In[ ]:





# In[ ]:


xgb = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.12,subsample=0.5)


# In[ ]:


y_log_train = np.log(y_train)
y_log_val = np.log(y_val)


# In[ ]:


xgb.fit(X_train, y_log_train, eval_set=[(X_val, y_log_val)], verbose=True)


# In[ ]:


y_pred = np.exp(xgb.predict(X_test))


# In[ ]:


res = pd.DataFrame()
res['Id'] = dataFrame_test['Id']
res['SalePrice'] = y_pred
res.to_csv('rf.csv', index=False)


# In[ ]:


#https://towardsdatascience.com/why-automated-feature-engineering-will-change-the-way-you-do-machine-learning-5c15bf188b96


# In[ ]:




