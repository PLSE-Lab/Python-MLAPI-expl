#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
from random import randint
from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# In[ ]:


train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.shape, test.shape


# # 1. Data Preprocessing

# In[ ]:


data = pd.concat([train, test], ignore_index=True, sort=False)
data.sample(5)


# ## 1-1. Type Issue

# In[ ]:


data['MSSubClass'] = data['MSSubClass'].astype(str)


# In[ ]:


num = data.select_dtypes(include=['int', 'float'])
cat = data.select_dtypes(include='object')

num.shape, cat.shape


# ## 1-2. Missing Value Replacement

# In[ ]:


num.isna().sum().sort_values(ascending=False)[:15]


# In[ ]:


# labels = data[['Id', 'SalePrice']]
# data.drop(columns=['SalePrice'], inplace=True)

num_feat = list(data.dtypes[data.dtypes != 'object'].index)
num_feat.remove('SalePrice')
cat_feat = data.dtypes[data.dtypes == 'object'].index

data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].mean())
data[num_feat] = data[num_feat].fillna(0)

data[num_feat].isna().sum().sort_values(ascending=False)


# In[ ]:


cat.isna().sum().sort_values(ascending=False)[:25]


# In[ ]:


for col in ('Alley','Utilities','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
            'BsmtFinType2','Electrical','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',
           'PoolQC','Fence','MiscFeature'):
    data[col]=data[col].fillna('None')
    
for col in ('MSZoning','Exterior1st','Exterior2nd','KitchenQual','SaleType','Functional'):
    data[col]=data[col].fillna(data[col].mode()[0])
    
data.isna().sum().sort_values(ascending=False)[:5]


# ## 1-3. Handling Skewness

# In[ ]:


skewed = [col for col in num_feat if np.abs(data[col].skew())>0.7]
skewed


# In[ ]:


lam = 0.15
data[skewed] = boxcox1p(data[skewed], lam)


# In[ ]:


data[num_feat].skew()


# In[ ]:


train_df = data.dropna()
test_df = data.drop(index=train_df.index)


# ## 1-4. Handling Outliers

# In[ ]:


train_df['Log_SalePrice'] = np.log1p(train_df['SalePrice'])


# In[ ]:


def scatter_machine(col, data=train_df):
    plt.figure(figsize=(12,7.5))
    plt.scatter(data[col], data['Log_SalePrice'])
    plt.title("%s Distribution" % col)


# In[ ]:


scatter_machine(col='GrLivArea')
plt.axvline(17)


# In[ ]:


train_df = train_df[train_df['GrLivArea']<17]
scatter_machine(col='LotArea')
plt.axvline(30)


# In[ ]:


train_df = train_df[train_df['LotArea']<30]
scatter_machine(col='PoolArea')


# In[ ]:


scatter_machine(col='KitchenAbvGr')


# In[ ]:


scatter_machine(col='TotRmsAbvGrd')


# ## 1-5. Scaling and Encoding

# In[ ]:


labels = train_df['SalePrice']
features = pd.concat([train_df, test_df], sort=False).drop(columns=['SalePrice', 'Log_SalePrice'])


# In[ ]:


scaler = RobustScaler()
features[num_feat] = scaler.fit_transform(features[num_feat])
features.head()


# In[ ]:


features = pd.get_dummies(features)
features.drop(columns=['Id'])
features.shape


# In[ ]:


labels = np.log1p(labels)


# # 2. Modeling

# In[ ]:


X_train = features.iloc[:len(train_df)]
X_test = features.drop(index=X_train.index)
y = labels
len(X_train), len(X_test), len(y)


# In[ ]:


lasso = Lasso()
enet = ElasticNet()
ridge = Ridge()
rfr = RandomForestRegressor()
gbr = GradientBoostingRegressor()

import xgboost as xgb
xgbr = xgb.XGBRegressor()


# In[ ]:


import warnings
warnings.filterwarnings('ignore')
models = [lasso, enet, ridge, rfr, gbr, xgbr]
scores = []
for model in models:
#     model.fit(X_train, y)
    means = cross_val_score(model, X_train, y, cv=5).mean()
    scores.append(means)
scores


# In[ ]:


from sklearn.model_selection import KFold

n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train.values)
    rmse= np.sqrt(-cross_val_score(model, X_train.values, y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

models = [lasso, enet, ridge, rfr, gbr, xgbr]
# rmsle_cv(ridge)
for model in models:
    score = rmsle_cv(model)
    print(score.mean())


# # 3. Submission

# In[ ]:


ridge.fit(X_train, y) # 0.012
rfr.fit(X_train, y)   # 0.015
gbr.fit(X_train, y)   # 0.012
xgbr.fit(X_train, y)  # 0.012
pred1 = ridge.predict(X_test)
pred2 = rfr.predict(X_test)
pred3 = gbr.predict(X_test)
pred4 = xgbr.predict(X_test)
prediction = 0.27*pred1 + 0.19*pred2 + 0.27*pred3 + 0.27*pred4

test['SalePrice'] = np.expm1(prediction)
submission = test[['Id', 'SalePrice']]
submission.to_csv('submission7.csv', index=False)


# In[ ]:


submission.head()


# In[ ]:




