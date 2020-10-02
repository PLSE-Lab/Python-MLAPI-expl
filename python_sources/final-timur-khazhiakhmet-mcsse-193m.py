#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# > Data preparation

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")


# In[ ]:


train.info()


# > Feature selection and data cleaning

# In[ ]:


train.drop(columns=['Id'], inplace=True)


# In[ ]:


import seaborn as sns
plt.figure(figsize=(20,20))
g = sns.heatmap(train.corr(), annot=True, cmap="RdYlGn")


# In[ ]:


train.isnull().sum().sort_values(ascending=False)[:30]


# In[ ]:


has_too_much_null_columns = [
    'PoolQC',
    'MiscFeature',
    'Alley',
    'Fence'
]

train.drop(columns = has_too_much_null_columns, inplace=True)


# In[ ]:


has_null_columns = [
    'FireplaceQu',
    'LotFrontage',
    'GarageCond',
    'GarageType',
    'GarageYrBlt',
    'GarageFinish',
    'GarageQual',
    'BsmtExposure',
    'BsmtFinType2',
    'BsmtFinType1',
    'BsmtCond',
    'BsmtQual',
    'MasVnrArea',
    'MasVnrType',
    'Electrical'
];

for col in has_null_columns:
    if (train[col].dtype == np.object):
        train[col].fillna(0, inplace=True)
    else:
        train[col].fillna(train[col].median(), inplace=True)


# > Data preprocessing

# In[ ]:


object_columns = list(train.select_dtypes(include=['object']).columns)


# In[ ]:


train_encoded = pd.get_dummies(train.iloc[:, :-1], columns=object_columns)


# In[ ]:


X = train_encoded.iloc[:, :-1]
y = train.iloc[:, -1]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2020)


# > Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
y_preds_lr = lr.predict(X_test)


# In[ ]:


from sklearn.metrics import r2_score
r2_score(y_test, y_preds_lr)


# In[ ]:


def adj_r2(r2score, train):
    return (1 - (1 - r2score) * ((train.shape[0] - 1) / (train.shape[0] - train.shape[1] - 1)))

adj_r2(r2_score(y_test, y_preds_lr), X_train)


# In[ ]:


import statsmodels.api as sm
regressor_OLS = sm.OLS(endog = y, exog = X).fit()
regressor_OLS.summary()


# In[ ]:


def comparing_preds_and_test(y_test, y_preds):
    plt.scatter(y_test, y_preds)
    plt.xlabel('y_test')                       
    plt.ylabel('y_preds_lr')
    plt.show()

comparing_preds_and_test(y_test, y_preds_lr)


# > SVR

# In[ ]:


from sklearn.svm import SVR

svr = SVR()
svr.fit(X_train, y_train)
y_preds_svr = svr.predict(X_test)


# In[ ]:


r2_score(y_test, y_preds_svr), adj_r2(r2_score(y_test, y_preds_svr), X_train)


# In[ ]:


comparing_preds_and_test(y_test, y_preds_svr)


# > Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=2020)
rf.fit(X_train, y_train)
y_preds_rf = rf.predict(X_test)


# In[ ]:


r2_score(y_test, y_preds_rf), adj_r2(r2_score(y_test, y_preds_rf), X_train)


# In[ ]:


comparing_preds_and_test(y_test, y_preds_rf)


# > Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=2020)
dt.fit(X_train,y_train)
y_preds_dt = dt.predict(X_test)


# In[ ]:


r2_score(y_test, y_preds_dt), adj_r2(r2_score(y_test, y_preds_dt), X_train)


# In[ ]:


comparing_preds_and_test(y_test, y_preds_dt)


# > XGB

# In[ ]:


from xgboost.sklearn import XGBRegressor

xgb = XGBRegressor()
xgb.fit(X_train, y_train)
y_preds_xgb = xgb.predict(X_test)


# In[ ]:


r2_score(y_test, y_preds_xgb), adj_r2(r2_score(y_test, y_preds_xgb), X_train)


# In[ ]:


comparing_preds_and_test(y_test, y_preds_xgb)


# > Submitting results

# In[ ]:


test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")


# In[ ]:


test.head()


# In[ ]:


test.drop(columns=has_too_much_null_columns, inplace=True)


# In[ ]:


test_has_null_columns = test.isnull().sum().sort_values(ascending=False)[:30]
test_has_null_columns


# In[ ]:


test_has_null_columns = [
    'FireplaceQu',
    'LotFrontage',
    'GarageCond',
    'GarageQual',
    'GarageYrBlt',
    'GarageFinish',
    'GarageType',
    'BsmtCond',
    'BsmtQual',
    'BsmtExposure',
    'BsmtFinType1',
    'BsmtFinType2',
    'MasVnrType',
    'MasVnrArea',
    'MSZoning',
    'BsmtHalfBath',
    'Utilities',
    'Functional',
    'BsmtFullBath',
    'BsmtUnfSF',
    'SaleType',
    'BsmtFinSF2',
    'BsmtFinSF1',
    'Exterior2nd',
    'Exterior1st',
    'TotalBsmtSF',
    'GarageCars',
    'KitchenQual',
    'GarageArea'
]

for col in test_has_null_columns:
    if (test[col].dtype == np.object):
        test[col].fillna(0, inplace=True)
    else:
        test[col].fillna(test[col].median(), inplace=True)


# In[ ]:


test_object_columns = list(test.select_dtypes(include=['object']).columns)
test_encoded = pd.get_dummies(test, columns=test_object_columns)
test_encoded


# In[ ]:


for col in test_encoded.columns:
    if (col not in X.columns):
        test_encoded.drop(columns=[col], inplace=True)

X.shape, test_encoded.shape


# In[ ]:


for col in X.columns:
    if (col not in test_encoded.columns):
        test_encoded[col] = 0
        
X.shape, test_encoded.shape


# In[ ]:


y_preds_lr_res = lr.predict(test_encoded)
y_preds_svr_res = svr.predict(test_encoded)
y_preds_rf_res = rf.predict(test_encoded)
y_preds_dt_res = dt.predict(test_encoded)


# In[ ]:


i = 0
rows_list = []
for pred in y_preds_lr_res:
    row = {'Id': test["Id"][i], 'SalePrice': pred}
    i += 1
    rows_list.append(row)
df = pd.DataFrame(rows_list) 
df.to_csv("y_preds_lr_res.csv", index=False)

i = 0
rows_list = []
for pred in y_preds_svr_res:
    row = {'Id': test["Id"][i], 'SalePrice': pred}
    i += 1
    rows_list.append(row)
df = pd.DataFrame(rows_list) 
df.to_csv("y_preds_svr_res.csv", index=False)

i = 0
rows_list = []
for pred in y_preds_rf_res:
    row = {'Id': test["Id"][i], 'SalePrice': pred}
    i += 1
    rows_list.append(row)
df = pd.DataFrame(rows_list) 
df.to_csv("y_preds_rf_res.csv", index=False)

i = 0
rows_list = []
for pred in y_preds_dt_res:
    row = {'Id': test["Id"][i], 'SalePrice': pred}
    i += 1
    rows_list.append(row)
df = pd.DataFrame(rows_list) 
df.to_csv("y_preds_dt_res.csv", index=False)

