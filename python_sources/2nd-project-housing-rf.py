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


# # 1. Start the project by doing exploratory data analysis

# In[ ]:


df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')


# In[ ]:


df.info()


# In[ ]:


df.head(2)


# In[ ]:


df.describe()


# # 2. Look for missing values

# In[ ]:


df.isnull().sum()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


plt.figure(figsize = (20,6))
sns.heatmap(df.isnull(), cbar = False)


# In[ ]:


df.columns[df.isnull().mean() > 0.2]


# # 2.1 Delete columns with missing values over 20%

# In[ ]:


df.drop(['Id','Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis = 1, inplace = True)


# In[ ]:


df.head(2)


# In[ ]:


plt.figure(figsize = (20,6))
sns.heatmap(df.isnull(), cbar = False);


# In[ ]:


df.LotFrontage.isnull().sum()/len(df.LotFrontage)


# # 2.2 Look for correlation

# In[ ]:


corr = df.corr()


# In[ ]:


plt.figure(figsize=(30,15))
sns.heatmap(corr, square=True, vmin = -1, vmax = 1,cmap = 'coolwarm', linewidths=.5);


# # 2.3 Treat missing values

# In[ ]:


df.LotFrontage.fillna(df.LotFrontage.mean(), inplace = True)


# In[ ]:


plt.figure(figsize = (20,6))
sns.heatmap(df.isnull(), cbar = False);


# In[ ]:


df.columns[df.isnull().sum() > 0]


# In[ ]:


df[['MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
       'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']].isnull().sum()/len(df[['MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
       'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']])


# In[ ]:


df.dropna(inplace = True)


# In[ ]:


df


# In[ ]:


df.select_dtypes('object').columns


# # 3 Create dummy variables

# In[ ]:


df1 = pd.get_dummies(df, columns=['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
       'PavedDrive', 'SaleType', 'SaleCondition'], drop_first=True)


# In[ ]:


print(df.columns.values)


# In[ ]:


print(df1.columns.values)


# # 4. Delete strong correlated columns

# In[ ]:


corr_matrix = df1.corr().abs()


# In[ ]:


upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))


# In[ ]:


to_drop = [column for column in upper.columns if any(upper[column] > 0.70)]


# In[ ]:


to_drop.remove('SalePrice')


# In[ ]:


to_drop


# In[ ]:


df1.head(2)


# In[ ]:





# In[ ]:


train = df1.drop(to_drop,axis=1)


# In[ ]:


train.columns.values


# # 5. Partition Data

# In[ ]:


from sklearn.model_selection import train_test_split
X = train.drop('SalePrice', axis = 1)
y = train.SalePrice


# # 6. Build Model

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


RF = RandomForestRegressor()


# In[ ]:


RF.fit(X,y)


# # 7. Handle Test Data

# In[ ]:


test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


test.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis = 1, inplace = True)


# In[ ]:


test.LotFrontage.fillna(test.LotFrontage.mean(), inplace = True)


# In[ ]:


test.columns[test.isnull().sum() > 0]


# In[ ]:


test.BsmtFullBath.dtype


# In[ ]:


test.GarageFinish.isnull().sum()


# In[ ]:


test.SaleType.value_counts()


# In[ ]:


test.MSZoning.fillna('RL', inplace = True)
test.Utilities.fillna('AllPub', inplace = True)
test.Exterior1st.fillna('VinylSd', inplace = True)
test.Exterior2nd.fillna('VinylSd', inplace = True)
test.MasVnrType.fillna('None', inplace = True)
test.MasVnrArea.fillna(test.MasVnrArea.mean(), inplace = True)
test.BsmtQual.fillna('TA', inplace = True)
test.BsmtCond.fillna('TA', inplace = True)
test.BsmtExposure.fillna('No', inplace = True)
test.BsmtFinType1.fillna('GLQ', inplace = True)
test.BsmtFinSF1.fillna(test.BsmtFinSF1.mean(), inplace = True)
test.BsmtFinType2.fillna('Unf', inplace = True) 
test.BsmtFinSF2.fillna(test.BsmtFinSF2.mean(), inplace = True)
test.BsmtUnfSF.fillna(test.BsmtUnfSF.mean(), inplace = True)
test.TotalBsmtSF.fillna(test.TotalBsmtSF.mean(), inplace = True)
test.BsmtFullBath.fillna(0.0, inplace = True)
test.BsmtHalfBath.fillna(0.0, inplace = True)
test.KitchenQual.fillna('TA', inplace = True) 
test.Functional.fillna('Typ', inplace = True)
test.GarageType.fillna('Attchd', inplace = True)
test.GarageYrBlt.fillna(2005, inplace = True)
test.GarageFinish.fillna('Unf', inplace = True) 
test.GarageCars.fillna(2, inplace = True)
test.GarageArea.fillna(test.GarageArea.mean(), inplace = True)
test.GarageQual.fillna('TA', inplace = True)
test.GarageCond.fillna('TA', inplace = True)
test.SaleType.fillna('WD', inplace = True)


# In[ ]:


test.info()


# In[ ]:


test.head(2)


# In[ ]:


test1 = pd.get_dummies(test, columns=['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
       'PavedDrive', 'SaleType', 'SaleCondition'], drop_first=True)


# In[ ]:


test1.head(2)


# In[ ]:


to_drop


# In[ ]:


missing_cols = set( train.columns ) - set( test.columns )


# In[ ]:


extra_cols = set( test1.columns ) - set( train.columns )


# In[ ]:


extra_cols


# In[ ]:


test2 = test1.drop(extra_cols, axis = 1)


# In[ ]:


test2


# In[ ]:


missing_cols = set( train.columns ) - set( test2.columns )


# In[ ]:


missing_cols


# In[ ]:


for c in missing_cols:
    test2[c] = 0


# In[ ]:


test_final = test2[train.columns]


# In[ ]:


test_final.columns.values


# In[ ]:


train.columns.values


# In[ ]:


test_final = test_final.drop('SalePrice', axis =1)


# # 8. make predictions

# In[ ]:


predictions = RF.predict(test_final)


# # 9. Output

# In[ ]:


outputRF = pd.DataFrame({'Id': test.Id, 'SalePrice': predictions})


# In[ ]:


outputRF.to_csv('house_submission2.csv', index=False) 

