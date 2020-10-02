#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### Import required libraries

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import ensemble
from sklearn import model_selection
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer, mean_squared_error

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler

from IPython.display import display # Allows the use of display() for DataFrames

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#### Routine to Fill Basic Missing Data - Regressor
def fill_missing_data_basic_reg(X_train, y_train, X_test):
    rf_reg_est = ensemble.RandomForestRegressor(random_state = 42)
    rf_reg_est.fit(X_train, y_train)
    y_test = rf_reg_est.predict(X_test)
    
    return y_test

#### Routine to Fill Basic Missing Data - Classifier
def fill_missing_data_basic_class(X_train, y_train, X_test):
    rf_class_est = ensemble.RandomForestClassifier(random_state = 42)
    rf_class_est.fit(X_train, y_train)
    y_test = rf_class_est.predict(X_test)
    
    return y_test


# In[ ]:


### Read Train and Test Data

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# In[ ]:


### Analyse Train Data

print("Shape of Train Data -> ".format(train_df.shape))
train_df.info()
train_df.head()


# In[ ]:


train_df.describe()


# In[ ]:


#### Check if there are any NULL values in Train Data
print("Total Train Features with NaN Values = " + str(train_df.columns[train_df.isnull().sum() != 0].size))
print("Features with NaN => {}".format(list(train_df.columns[train_df.isnull().sum() != 0])))
train_df[train_df.columns[train_df.isnull().sum() != 0]].isnull().sum().sort_values(ascending = False)


# In[ ]:


### Analyse Test Data

print("Shape of Test Data -> " + str(test_df.shape))
test_df.info()
test_df.head()


# In[ ]:


test_df.describe()


# In[ ]:


#### Check if there are any NULL values in Test Data
print("Total Test Features with NaN Values = " + str(test_df.columns[test_df.isnull().sum() != 0].size))
print("Features with NaN => {}".format(list(test_df.columns[test_df.isnull().sum() != 0])))
test_df[test_df.columns[test_df.isnull().sum() != 0]].isnull().sum().sort_values(ascending = False)


# In[ ]:


### Combine Train and Test Data
test_df['SalePrice'] = 0
combined_train_test_df = train_df.append(test_df)

print("Shape of Combined Train and Test Data -> " + str(combined_train_test_df.shape))


# In[ ]:


#### Check if there are any NULL values in Train and Test Data Combined
print("Total Features with NaN Values in Test and Train = " + str(combined_train_test_df.columns[combined_train_test_df.isnull().sum() != 0].size))
combined_train_test_df.columns[combined_train_test_df.isnull().sum() != 0]
missing_total = combined_train_test_df[combined_train_test_df.columns[combined_train_test_df.isnull().sum() != 0]].isnull().sum().sort_values(ascending = False)
missing_percent = (combined_train_test_df[combined_train_test_df.columns[combined_train_test_df.isnull().sum() != 0]].isnull().sum()/combined_train_test_df.shape[0] * 100).sort_values(ascending = False)


# In[ ]:


missing_features = missing_total.index 
missing_features_tot = pd.concat([missing_total, missing_percent], axis = 1)
print(missing_features_tot)


# In[ ]:


missing_data_combined_df = combined_train_test_df[combined_train_test_df.columns[combined_train_test_df.isnull().sum() != 0]]
missing_data_combined_df.info()
missing_data_combined_df.describe()


# In[ ]:


# Print all the 'Numerical' columns
print("Numerical Columns -> {}".format(list(combined_train_test_df.select_dtypes(include=[np.number]).columns)))


# In[ ]:


# Print all the 'Categorical' columns
print("Categorical Columns -> {}".format(list(combined_train_test_df.select_dtypes(exclude=[np.number]).columns)))


# In[ ]:


### Analyze SalePrice
train_df['SalePrice'].describe()


# In[ ]:


plt.figure(figsize=(12, 6))
_ = sns.distplot(train_df['SalePrice'])


# In[ ]:


plt.figure(figsize=(12, 4))
_ = sns.boxplot(train_df['SalePrice'])


# In[ ]:


train_num_df = train_df.select_dtypes(include=[np.number])
train_num_df = train_num_df.drop(['Id'], axis=1)

fig, axs = plt.subplots(12,3, figsize=(16, 30), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.2, right=0.95)

axs = axs.ravel()

for ind, col in enumerate(train_num_df.columns):
    if col != 'SalePrice':
        sns.regplot(train_num_df[col], train_num_df['SalePrice'], ax = axs[ind])
    
plt.show()


# In[ ]:


train_cat_df = train_df.select_dtypes(exclude=[np.number])
train_cat_df['SalePrice'] = train_df['SalePrice']

fig, axs = plt.subplots(15,3, figsize=(16, 30), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.2, right=0.95)

axs = axs.ravel()

for ind, col in enumerate(train_cat_df.columns):
    if col != 'SalePrice':
        sns.barplot(train_cat_df[col], train_cat_df['SalePrice'], ax = axs[ind])

plt.show()


# In[ ]:


# Display the correlation heatmap
corr = train_df.drop(['Id'], axis=1).corr()

plt.figure(figsize = (16,10))
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax1 = sns.heatmap(corr, mask=mask)


# In[ ]:


corr.sort_values(["SalePrice"], ascending = False, inplace = True)
print(corr.SalePrice)


# In[ ]:


#### BsmtUnfSF
if (combined_train_test_df['BsmtUnfSF'].isnull().sum() != 0):
    combined_train_test_df['BsmtUnfSF'].fillna(round(combined_train_test_df['BsmtUnfSF'].mean()), inplace = True)


# In[ ]:


#### BsmtFullBath
if (combined_train_test_df['BsmtFullBath'].isnull().sum() != 0):
    combined_train_test_df['BsmtFullBath'].fillna(combined_train_test_df['BsmtFullBath'].mode().iloc[0], inplace = True)


# In[ ]:


#### BsmtHalfBath
if (combined_train_test_df['BsmtHalfBath'].isnull().sum() != 0):
    combined_train_test_df['BsmtHalfBath'].fillna(combined_train_test_df['BsmtHalfBath'].mode().iloc[0], inplace = True)


# In[ ]:


##### TotalBsmtSF
if (combined_train_test_df['TotalBsmtSF'].isnull().sum() != 0):
    combined_train_test_df['TotalBsmtSF'].fillna(round(combined_train_test_df['TotalBsmtSF'].mean()), inplace = True)


# In[ ]:


#### GarageArea
if (combined_train_test_df['GarageArea'].isnull().sum() != 0):
    combined_train_test_df['GarageArea'].fillna(round(combined_train_test_df['GarageArea'].mean()), inplace = True)


# In[ ]:


#### GarageCars
if (combined_train_test_df['GarageCars'].isnull().sum() != 0):
    combined_train_test_df['GarageCars'].fillna(combined_train_test_df['GarageCars'].mode().iloc[0], inplace = True)


# In[ ]:


#### KitchenQual
if (combined_train_test_df['KitchenQual'].isnull().sum() != 0):
    combined_train_test_df['KitchenQual'].fillna(combined_train_test_df['KitchenQual'].mode().iloc[0], inplace = True)


# In[ ]:


#### Electrical
if (combined_train_test_df['Electrical'].isnull().sum() != 0):
    combined_train_test_df['Electrical'].fillna(combined_train_test_df['Electrical'].mode().iloc[0], inplace = True)


# In[ ]:


#### BsmtFinSF2
if (combined_train_test_df['BsmtFinSF2'].isnull().sum() != 0):
    combined_train_test_df['BsmtFinSF2'].fillna(round(combined_train_test_df['BsmtFinSF2'].mean()), inplace = True)


# In[ ]:


#### BsmtFinSF1
if (combined_train_test_df['BsmtFinSF1'].isnull().sum() != 0):
    combined_train_test_df['BsmtFinSF1'].fillna(round(combined_train_test_df['BsmtFinSF1'].mean()), inplace = True)


# In[ ]:


#### SaleType
if (combined_train_test_df['SaleType'].isnull().sum() != 0):
    combined_train_test_df['SaleType'].fillna(combined_train_test_df['SaleType'].mode().iloc[0], inplace = True)


# In[ ]:


#### Exterior1st
if (combined_train_test_df['Exterior1st'].isnull().sum() != 0):
    combined_train_test_df['Exterior1st'].fillna(combined_train_test_df['Exterior1st'].mode().iloc[0], inplace = True)


# In[ ]:


#### Exterior2nd
if (combined_train_test_df['Exterior2nd'].isnull().sum() != 0):
    combined_train_test_df['Exterior2nd'].fillna(combined_train_test_df['Exterior2nd'].mode().iloc[0], inplace = True)


# In[ ]:


#### Functional
if (combined_train_test_df['Functional'].isnull().sum() != 0):
    combined_train_test_df['Functional'].fillna(combined_train_test_df['Functional'].mode().iloc[0], inplace = True)


# In[ ]:


#### Utilities
if (combined_train_test_df['Utilities'].isnull().sum() != 0):
    combined_train_test_df['Utilities'].fillna(combined_train_test_df['Utilities'].mode().iloc[0], inplace = True)


# In[ ]:


#### MSZoning
if (combined_train_test_df['MSZoning'].isnull().sum() != 0):
    combined_train_test_df['MSZoning'].fillna(combined_train_test_df['MSZoning'].mode().iloc[0], inplace = True)


# In[ ]:


#### MasVnrArea
if (combined_train_test_df['MasVnrArea'].isnull().sum() != 0):
    combined_train_test_df['MasVnrArea'].fillna(0, inplace = True)


# In[ ]:


#### MasVnrType
if (combined_train_test_df['MasVnrType'].isnull().sum() != 0):
    combined_train_test_df['MasVnrType'].fillna('None', inplace = True)


# In[ ]:


#### BsmtCond
if (combined_train_test_df['BsmtCond'].isnull().sum() != 0):
    combined_train_test_df['BsmtCond'].fillna(combined_train_test_df['BsmtCond'].mode().iloc[0], inplace = True)


# In[ ]:


#### BsmtExposure
if (combined_train_test_df['BsmtExposure'].isnull().sum() != 0):
    combined_train_test_df['BsmtExposure'].fillna(combined_train_test_df['BsmtExposure'].mode().iloc[0], inplace = True)


# In[ ]:


#### BsmtQual
if (combined_train_test_df['BsmtQual'].isnull().sum() != 0):
    combined_train_test_df['BsmtQual'].fillna(combined_train_test_df['BsmtQual'].mode().iloc[0], inplace = True)


# In[ ]:


#### BsmtFinType1
if (combined_train_test_df['BsmtFinType1'].isnull().sum() != 0):
    combined_train_test_df['BsmtFinType1'].fillna(combined_train_test_df['BsmtFinType1'].mode().iloc[0], inplace = True)


# In[ ]:


#### BsmtFinType2
if (combined_train_test_df['BsmtFinType2'].isnull().sum() != 0):
    combined_train_test_df['BsmtFinType2'].fillna(combined_train_test_df['BsmtFinType2'].mode().iloc[0], inplace = True)


# In[ ]:


#### GarageType
if (combined_train_test_df['GarageType'].isnull().sum() != 0):
    combined_train_test_df['GarageType'].fillna(combined_train_test_df['GarageType'].mode().iloc[0], inplace = True)


# In[ ]:


#### GarageCond
if (combined_train_test_df['GarageCond'].isnull().sum() != 0):
    combined_train_test_df['GarageCond'].fillna(combined_train_test_df['GarageCond'].mode().iloc[0], inplace = True)


# In[ ]:


#### GarageQual
if (combined_train_test_df['GarageQual'].isnull().sum() != 0):
    combined_train_test_df['GarageQual'].fillna(combined_train_test_df['GarageQual'].mode().iloc[0], inplace = True)


# In[ ]:


#### GarageFinish
if (combined_train_test_df['GarageFinish'].isnull().sum() != 0):
    combined_train_test_df['GarageFinish'].fillna(combined_train_test_df['GarageFinish'].mode().iloc[0], inplace = True)


# In[ ]:


#### GarageYrBlt
if (combined_train_test_df['GarageYrBlt'].isnull().sum() != 0):
    combined_train_test_df['GarageYrBlt'].fillna(combined_train_test_df['YearBuilt'], inplace = True)


# In[ ]:


#### LotFrontage
if (combined_train_test_df['LotFrontage'].isnull().sum() != 0):
    df_LF_LA = combined_train_test_df[['LotFrontage', 'LotArea']]
    X_train_LF_LA = df_LF_LA[df_LF_LA['LotFrontage'].notnull()].drop(labels = ('LotFrontage'), axis = 1)
    y_train_LF_LA = df_LF_LA[df_LF_LA['LotFrontage'].notnull()].drop(labels = ['LotArea'], axis = 1)
    
    X_test_LF_LA = df_LF_LA[df_LF_LA['LotFrontage'].isnull()].drop(['LotFrontage'], axis = 1)
    y_test_LF_LA = fill_missing_data_basic_reg(X_train_LF_LA, y_train_LF_LA, X_test_LF_LA)
    
    combined_train_test_df.loc[(combined_train_test_df.LotFrontage.isnull()), 'LotFrontage'] = y_test_LF_LA


# In[ ]:


#### FireplaceQu
if (combined_train_test_df['FireplaceQu'].isnull().sum().sum() != 0):
    combined_train_test_df['FireplaceQu'] = combined_train_test_df['FireplaceQu'].fillna('NoFirePlace')


# In[ ]:


#### Fence
if (combined_train_test_df['Fence'].isnull().sum().sum() != 0):
    combined_train_test_df['Fence'] = combined_train_test_df['Fence'].fillna('NoFence')


# In[ ]:


#### Alley
if (combined_train_test_df['Alley'].isnull().sum().sum() != 0):
    combined_train_test_df['Alley'] = combined_train_test_df['Alley'].fillna('NoAlley')


# In[ ]:


#### MiscFeature
if (combined_train_test_df['MiscFeature'].isnull().sum().sum() != 0):
    combined_train_test_df['MiscFeature'] = combined_train_test_df['MiscFeature'].fillna('NoMiscFeature')


# In[ ]:


#### PoolQC
if (combined_train_test_df['PoolQC'].isnull().sum().sum() != 0):
    combined_train_test_df['PoolQC'] = combined_train_test_df['PoolQC'].fillna('NoPoolQC')


# In[ ]:


assert (combined_train_test_df.columns[combined_train_test_df.isnull().sum() != 0].size == 0)
print("Total Features with NaN in Test and Train After Imputation = " + str(combined_train_test_df.columns[combined_train_test_df.isnull().sum() != 0].size))


# In[ ]:


category_cols = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
continuos_cols = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageYrBlt', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SalePrice']


# In[ ]:


# log Transformation
cmb_log_df = combined_train_test_df.select_dtypes(include=[np.number])
cmb_log_df = cmb_log_df.drop(['Id', 'SalePrice'], axis=1)

for col in cmb_log_df.columns:
    cmb_log_df[col] = np.log1p(cmb_log_df[col])


# In[ ]:


# Normalize data
rob_sca = RobustScaler()
rob_sca.fit(cmb_log_df)
cmb_sca_df = pd.DataFrame(rob_sca.transform(cmb_log_df), index=cmb_log_df.index, columns=cmb_log_df.columns)


# In[ ]:


category_cols_dummies_df = pd.get_dummies(combined_train_test_df[category_cols])
cmb_sca_df = pd.concat([cmb_sca_df, category_cols_dummies_df], axis = 1)


# In[ ]:


#### Divide the Train and Test data
train_data = cmb_sca_df[:1460]
test_data = cmb_sca_df[1460:]


# In[ ]:


# SalePrice
train_data['SalePrice'] = np.log1p(train_df['SalePrice'])


# In[ ]:


train_data_X = train_data.drop(['SalePrice'], axis = 1)
train_data_y = train_data['SalePrice']

test_data_X = test_data


# In[ ]:


# RMSE
def RMSE(estimator,X_train, Y_train, cv=5,n_jobs=4):
    cv_results = cross_val_score(estimator,X_train,Y_train,cv=cv,scoring="neg_mean_squared_error",n_jobs=n_jobs)
    return (np.sqrt(-cv_results)).mean()


# In[ ]:


clf = LassoCV(random_state=42)
clf.fit(train_data_X, train_data_y)


# In[ ]:


#### Predict the output
df_pred = pd.DataFrame()
df_pred['SalePrice'] = clf.predict(test_data_X)
df_pred['SalePrice'] = np.expm1(df_pred['SalePrice'])

test_data_X['SalePrice'] = df_pred[['SalePrice']]

#### Prepare submission file
submission = pd.DataFrame({'Id': test_df.loc[:, 'Id'],
                           'SalePrice': test_data_X.loc[:, 'SalePrice']})
submission.to_csv("../working/submission.csv", index = False)

