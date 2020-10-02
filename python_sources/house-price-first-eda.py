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

import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


sns.set()


# In[ ]:


def ecdf(data):
    x = np.sort(data)
    y = np.arange(0, len(x),1) / len(x)
    return x, y


# In[ ]:


def plot_histogram_train_test(feature, bins=100):
    # train and test histogram
    plt.figure(figsize=(20, 3))
    plt.subplot(1,2,1)
    plt.title('Train distribution')
    plt.xlabel(feature)
    plt.hist(train_df[feature], bins=bins)
    plt.subplot(1,2,2)
    plt.hist(test_df[feature], bins=bins)
    plt.title('Test distribution')
    plt.xlabel(feature)
    
def plot_kde_train_test(feature):
    # train and test kde
    plt.figure(figsize=(20, 3))
    plt.subplot(1,2,1)
    plt.title('Train distribution')
    plt.xlabel(feature)
    sns.kdeplot(train_df[feature])
    plt.subplot(1,2,2)
    sns.kdeplot(test_df[feature])
    plt.title('Test distribution')
    plt.xlabel(feature)
    
def plot_ecdf_train_test(feature):
    # train and test ecdf
    plt.figure(figsize=(20, 3))
    plt.subplot(1,2,1)
    plt.title('Train distribution')
    plt.xlabel(feature)
    plt.scatter(*ecdf(train_df[feature]))
    plt.subplot(1,2,2)
    plt.scatter(*ecdf(test_df[feature]))
    plt.title('Test distribution')
    plt.xlabel(feature)
    
def count_plot_train_test(feature):
    # train and test countplot
    plt.figure(figsize=(20, 3))
    plt.subplot(1,2,1)
    plt.title('Train distribution')
    plt.xlabel(feature)
    sns.countplot(train_df[feature])
    plt.subplot(1,2,2)
    sns.countplot(test_df[feature])
    plt.title('Test distribution')
    plt.xlabel(feature)
    
def checknull(feature):
    print('Portion of null values in train and test respectively:', 
          train_df[feature].isnull().mean(), ',', 
          test_df[feature].isnull().mean())


# # 1. EDA

# In[ ]:


train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


test_df.info()


# In[ ]:


print('Number of columns in dataset:', len(train_df.columns))
print('Including 1 target columns: SalePrice')


# In[ ]:


# drop columns having null value > 30%
null_thresh = 0.3

is_nan_train = train_df.isnull().mean()
dropped_cols_train = is_nan_train[is_nan_train > null_thresh].index
is_nan_test = test_df.isnull().mean()
dropped_cols_test = is_nan_test[is_nan_test > null_thresh].index

print('Columns having null value > '+str(null_thresh),'in train_df:',dropped_cols_train)
print('Columns having null value > '+str(null_thresh),'in test_df:',dropped_cols_test)
# print('\nPortion of null of these columns in test_df:\n', test_df[dropped_cols].isnull().mean())
# train_df.drop(dropped_cols, axis=1, inplace=True)


# - So we decide to drop ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'] from both train and test set

# In[ ]:


train_df.drop(dropped_cols_train, axis=1, inplace=True)
test_df.drop(dropped_cols_test, axis=1, inplace=True)


# In[ ]:


train_df.info()


# - Choosing highly related columns with target variable

# In[ ]:


train_corr_matrix = train_df.corr()
valuable_cols = train_corr_matrix[np.abs(train_corr_matrix['SalePrice']) > 0.3].index
print('Columns having correlation with SalePrice > 0.3: ', valuable_cols.values)
train_df = train_df[valuable_cols]
test_df = test_df[valuable_cols[:-1]]


# In[ ]:


train_corr_matrix[np.abs(train_corr_matrix['SalePrice']) > 0.55].index


# In[ ]:


sns.heatmap(train_df.corr())


# In[ ]:


train_df.head()


# ## 1.1. LotFrontage:
# - Linear feet of street connected to property

# In[ ]:


train_df.LotFrontage.dtype


# In[ ]:


print('Portion of null values in train and test respectively:', train_df.LotFrontage.isnull().mean(), ',', test_df.LotFrontage.isnull().mean())


# In[ ]:


# train and test LotFrontage distribution
plt.figure(figsize=(20, 3))
plt.subplot(1,2,1)
plt.title('Train distribution')
plt.xlabel('LotFrontage')
sns.kdeplot(train_df['LotFrontage'])
plt.subplot(1,2,2)
sns.kdeplot(test_df['LotFrontage'])
plt.title('Test distribution')
plt.xlabel('LotFrontage')


# In[ ]:


# train and test LotFrontage histogram
plt.figure(figsize=(20, 3))
plt.subplot(1,2,1)
plt.title('Train distribution')
plt.xlabel('LotFrontage')
plt.hist(train_df['LotFrontage'], bins=100)
plt.subplot(1,2,2)
plt.hist(test_df['LotFrontage'], bins=100)
plt.title('Test distribution')
plt.xlabel('LotFrontage')


# In[ ]:


# train and test LotFrontage histogram
plt.figure(figsize=(13, 3))

plt.subplot(1,2,1)
plt.scatter(*ecdf(train_df['LotFrontage']))
plt.title('Train cumulative plot')
plt.xlabel('LotFrontage')

plt.subplot(1,2,2)
plt.scatter(*ecdf(test_df['LotFrontage']))
plt.title('Test cumulative plot')
plt.xlabel('LotFrontage')


# - There are a few outliers

# In[ ]:


# fir a regression line
plt.figure(figsize=(10,6))
sns.scatterplot('LotFrontage', 'SalePrice', data=train_df)
sns.regplot('LotFrontage', 'SalePrice', data=train_df, scatter=None)


# ## 1.2. OverallQual:
# OverallQual: Rates the overall material and finish of the house
# 
#        10	Very Excellent
#        9	Excellent
#        8	Very Good
#        7	Good
#        6	Above Average
#        5	Average
#        4	Below Average
#        3	Fair
#        2	Poor
#        1	Very Poor
# 	

# - It is an ordinal feature

# In[ ]:


train_df.OverallQual.dtype


# In[ ]:


print('Portion of null values in train and test respectively:', train_df.OverallQual.isnull().mean(), ',', test_df.OverallQual.isnull().mean())


# In[ ]:


plt.figure(figsize=(10, 5))

plt.subplot(1,2,1)
sns.countplot(train_df.OverallQual)
plt.title('Value count of each OverallQual type on train set')

plt.subplot(1,2,2)
sns.countplot(test_df.OverallQual)
plt.title('Value count of each OverallQual type on train set')


# In[ ]:


train_df.groupby('OverallQual').mean()['SalePrice']


# - We see the correlation between OverallQual with SalePrice on average of each quality types (1->10)

# In[ ]:


sns.scatterplot('OverallQual', 'SalePrice', data=train_df)


# ## 1.3. Year built:
# - Original construction date
# 

# In[ ]:


train_df.YearBuilt.dtype


# In[ ]:


print('Portion of null values in train and test respectively:', train_df.YearBuilt.isnull().mean(), ',', test_df.YearBuilt.isnull().mean())


# In[ ]:


np.sort(test_df.YearBuilt.unique())


# In[ ]:


np.sort(train_df.YearBuilt.unique())


# In[ ]:


# train and test histogram
plt.figure(figsize=(13, 3))

plt.subplot(1,2,1)
plt.scatter(*ecdf(train_df.YearBuilt))
plt.title('Train cumulative plot')
plt.xlabel('YearBuilt')

plt.subplot(1,2,2)
plt.scatter(*ecdf(test_df.YearBuilt))
plt.title('Test cumulative plot')
plt.xlabel('YearBuilt')


# In[ ]:


sns.scatterplot('YearBuilt', 'SalePrice', data=train_df)


# ## 1.4. YearRemodAdd:
# - Remodel date (same as construction date if no remodeling or additions)

# In[ ]:


train_df.YearRemodAdd.dtype


# In[ ]:


print('Portion of null values in train and test respectively:', train_df.YearRemodAdd.isnull().mean(), ',', test_df.YearRemodAdd.isnull().mean())


# In[ ]:


# train and test histogram
plt.figure(figsize=(13, 3))

plt.subplot(1,2,1)
plt.scatter(*ecdf(train_df.YearRemodAdd))
plt.title('Train cumulative plot')
plt.xlabel('YearRemodAdd')

plt.subplot(1,2,2)
plt.scatter(*ecdf(test_df.YearRemodAdd))
plt.title('Test cumulative plot')
plt.xlabel('YearRemodAdd')


# In[ ]:


sns.scatterplot('YearRemodAdd', 'SalePrice', data=train_df)


# In[ ]:


plt.scatter(train_df['YearRemodAdd']-train_df['YearBuilt'], train_df['SalePrice'])


# ## 1.5. MasVnrArea:
# - Masonry veneer area in square feet

# In[ ]:


train_df.MasVnrArea.dtype


# In[ ]:


print('Portion of null values in train and test respectively:', train_df.MasVnrArea.isnull().mean(), ',', test_df.MasVnrArea.isnull().mean())


# In[ ]:


# train and test histogram
plt.figure(figsize=(13, 3))

plt.subplot(1,2,1)
plt.scatter(*ecdf(train_df.MasVnrArea))
plt.title('Train cumulative plot')
plt.xlabel('MasVnrArea')

plt.subplot(1,2,2)
plt.scatter(*ecdf(test_df.MasVnrArea))
plt.title('Test cumulative plot')
plt.xlabel('MasVnrArea')


# In[ ]:


# train and test MasVnrArea histogram
plt.figure(figsize=(20, 3))
plt.subplot(1,2,1)
plt.title('Train distribution')
plt.xlabel('MasVnrArea')
plt.hist(train_df['MasVnrArea'], bins=100)
plt.subplot(1,2,2)
plt.hist(test_df['MasVnrArea'], bins=100)
plt.title('Test distribution')
plt.xlabel('MasVnrArea')


# - About 60% MasVnrArea of train set having value exactly 0. The same observation occur

# In[ ]:


(train_df.MasVnrArea == 0).sum()


# In[ ]:


(test_df.MasVnrArea == 0).sum()


# ## 1.6. BsmtFinSF1:
# - type 1 finished square feet

# In[ ]:


train_df.BsmtFinSF1.dtype


# In[ ]:


checknull('BsmtFinSF1')


# In[ ]:


plot_kde_train_test('BsmtFinSF1')


# In[ ]:


plot_histogram_train_test('BsmtFinSF1')


# In[ ]:


plot_ecdf_train_test('BsmtFinSF1')


# In[ ]:


sns.lmplot('BsmtFinSF1', 'SalePrice', data=train_df)


# ## 1.7. TotalBsmtSF:
# - Total square feet of basement area

# In[ ]:


train_df.TotalBsmtSF.dtype


# In[ ]:


checknull('TotalBsmtSF')


# In[ ]:


plot_histogram_train_test('TotalBsmtSF')


# In[ ]:


plot_kde_train_test('TotalBsmtSF')


# In[ ]:


plot_ecdf_train_test('TotalBsmtSF')


# In[ ]:


sns.lmplot('TotalBsmtSF', 'SalePrice', train_df)


# ## 1.8. 1stFlrSF:
# > - First Floor square feet

# In[ ]:


train_df['1stFlrSF'].dtype


# In[ ]:


checknull('1stFlrSF')


# In[ ]:


plot_histogram_train_test('1stFlrSF')


# In[ ]:


plot_kde_train_test('1stFlrSF')


# In[ ]:


plot_ecdf_train_test('1stFlrSF')


# In[ ]:


sns.lmplot('1stFlrSF', 'SalePrice', train_df)


# ## 1.9. 2ndFlrSF:
# - Second floor square feet

# In[ ]:


train_df['2ndFlrSF'].dtype


# In[ ]:


checknull('2ndFlrSF')


# In[ ]:


plot_histogram_train_test('2ndFlrSF')


# In[ ]:


plot_kde_train_test('2ndFlrSF')


# - 60% of both train and test set is exactly 0

# In[ ]:


plot_ecdf_train_test('2ndFlrSF')


# In[ ]:


sns.lmplot('2ndFlrSF', 'SalePrice', train_df)


# ## 1.10. GrLivArea:
# - Above grade (ground) living area square feet

# In[ ]:


train_df.GrLivArea.dtype


# In[ ]:


checknull('GrLivArea')


# In[ ]:


plot_histogram_train_test('GrLivArea')


# In[ ]:


plot_kde_train_test('GrLivArea')


# In[ ]:


plot_ecdf_train_test('GrLivArea')


# In[ ]:


sns.lmplot('GrLivArea', 'SalePrice', train_df)


# ## 1.10. FullBath:
# - Full bathrooms above grade
# - An ordinal feature

# In[ ]:


train_df.FullBath.dtype


# In[ ]:


checknull('FullBath')


# In[ ]:


count_plot_train_test('FullBath')


# In[ ]:


sns.lmplot('FullBath', 'SalePrice', train_df)


# ## 1.11. TotRmsAbvGrd:
# - Total rooms above grade (does not include bathrooms)

# In[ ]:


train_df.TotRmsAbvGrd.dtype


# In[ ]:


train_df.TotRmsAbvGrd.unique()


# In[ ]:


checknull('TotRmsAbvGrd')


# In[ ]:


count_plot_train_test('TotRmsAbvGrd')


# In[ ]:


sns.lmplot('TotRmsAbvGrd', 'SalePrice', data=train_df)


# ## 1.12. Fireplaces:
# - Number of fireplaces

# In[ ]:


train_df.Fireplaces.dtype


# In[ ]:


train_df.Fireplaces.unique()


# In[ ]:


checknull('Fireplaces')


# In[ ]:


count_plot_train_test('Fireplaces')


# In[ ]:


sns.lmplot('Fireplaces', 'SalePrice', train_df)


# ## 1.12. GarageYrBlt:
# - Year garage was built

# In[ ]:


train_df.GarageYrBlt.dtype


# In[ ]:


train_df.GarageYrBlt.unique()


# In[ ]:


plot_histogram_train_test('GarageYrBlt')


# In[ ]:


plot_kde_train_test('GarageYrBlt')


# In[ ]:


plot_ecdf_train_test('GarageYrBlt')


# In[ ]:


sns.lmplot('GarageYrBlt', 'SalePrice', train_df)


# ## 1.13. GarageCars:
# - Size of garage in car capacity

# In[ ]:


train_df.GarageCars.dtype


# In[ ]:


train_df.GarageCars.unique()


# In[ ]:


checknull('GarageCars')


# In[ ]:


count_plot_train_test('GarageCars')


# In[ ]:


sns.lmplot('GarageCars', 'SalePrice', train_df)


# ## 1.14. GarageArea:
# - Size of garage in square feet

# In[ ]:


train_df.GarageArea.dtype


# In[ ]:


checknull('GarageArea')


# In[ ]:


plot_histogram_train_test('GarageArea')


# In[ ]:


plot_kde_train_test('GarageArea')


# In[ ]:


plot_ecdf_train_test('GarageArea')


# In[ ]:


sns.lmplot('GarageArea', 'SalePrice', train_df)


# ## 1.15. WoodDeckSF:
# - Wood deck area in square feet

# In[ ]:


train_df.WoodDeckSF.dtype


# In[ ]:


checknull('WoodDeckSF')


# In[ ]:


plot_histogram_train_test('WoodDeckSF')


# In[ ]:


plot_kde_train_test('WoodDeckSF')


# In[ ]:


plot_ecdf_train_test('WoodDeckSF')


# In[ ]:


sns.lmplot('WoodDeckSF', 'SalePrice', train_df)


# ## 1.16. OpenPorchSF:
# - Open porch area in square feet

# In[ ]:


train_df.OpenPorchSF.dtype


# In[ ]:


checknull('OpenPorchSF')


# In[ ]:


plot_histogram_train_test('OpenPorchSF')


# In[ ]:


plot_kde_train_test('OpenPorchSF')


# In[ ]:


plot_ecdf_train_test('OpenPorchSF')


# In[ ]:


sns.lmplot('OpenPorchSF', 'SalePrice', train_df)


# # 2. Training

# In[ ]:


from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error


# In[ ]:


combine = [train_df, test_df]


# In[ ]:


feature_cols = ['LotFrontage', 'OverallQual', 'YearBuilt',
       'TotalBsmtSF', '1stFlrSF', 'GrLivArea',
       'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',
       'GarageArea']


# In[ ]:


train_df = train_df[feature_cols+['SalePrice']]
test_df = test_df[feature_cols]


# In[ ]:


# Impute columns with missing value
for col in feature_cols:
    feature_most_frequent_value_train = train_df[col].mode().values[0]
    train_df[col] = train_df[col].fillna(feature_most_frequent_value_train)
    test_df[col] = test_df[col].fillna(feature_most_frequent_value_train)


# In[ ]:


labels = train_df['SalePrice']
data = train_df.drop('SalePrice', axis=1)

mean_vec = np.mean(data, axis=0)
std_vec = np.std(data, axis=0)

data = (data - mean_vec) / std_vec

test_data = (test_df-mean_vec)/std_vec


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(data,
                                                   train_df['SalePrice'],
                                                   test_size=0.2,
                                                   random_state=123)


# In[ ]:


# params = {'C':[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]}

# grid = GridSearchCV(param_grid=params, estimator=LogisticRegression(multi_class='auto', solver='lbfgs'), cv=5, n_jobs=-1)


# In[ ]:


# grid.fit(X_train, y_train)


# In[ ]:


lr = LogisticRegression(multi_class='auto', solver='lbfgs')
# lr.fit(X_train, y_train)


# In[ ]:


dt = DecisionTreeRegressor(max_depth=9, min_samples_leaf=20)
# dt.fit(X_train, y_train)


# In[ ]:


rf = RandomForestRegressor(n_estimators=50, max_depth=9 ,min_samples_leaf=5)
# rf.fit(X_train, y_train)


# In[ ]:


gb = GradientBoostingRegressor(n_estimators=80)
gb.fit(X_train, y_train)


# In[ ]:


y_pred_train = gb.predict(X_train)
mean_squared_log_error(y_train, y_pred_train)**.5


# In[ ]:


y_pred_val = gb.predict(X_test)
mean_squared_log_error(y_test, y_pred_val)**.5


# In[ ]:


gb.fit(np.vstack([X_train, X_test]), np.concatenate([y_train, y_test]))
y_pred_test = gb.predict(test_data)


# In[ ]:


submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')


# In[ ]:


submission['SalePrice'] = y_pred_test


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:


submission.head()


# In[ ]:




