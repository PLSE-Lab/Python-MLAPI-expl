
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
train_dataset = pd.read_csv('../input/train.csv')
test_dataset = pd.read_csv('../input/test.csv')

################ Clean train_dataset
# Step 1: Check all columns for NaN values and fix/drop as necessary
from sklearn.preprocessing import Imputer

train_dataset.isnull().sum()
# Fix Alley -- Does property have alley: yes or no
train_dataset.groupby('Alley').Id.nunique()
train_dataset.ix[train_dataset.Alley.notnull(), 'Alley'] = 1
train_dataset.ix[train_dataset.Alley.isnull(), 'Alley'] = 0

# Fix LotFrontage -- change to median lot size
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer = imputer.fit(train_dataset[['LotFrontage']])
train_dataset[['LotFrontage']] = imputer.transform(train_dataset[['LotFrontage']])

# Fix MasVnrType -- updated the 8 null fields with 'None' since 'None' is majority of dataset 
# Fix MasVnrArea -- updated with 0 since 'MasVnrType' set to 'None'
train_dataset.groupby('MasVnrType').Id.nunique()
train_dataset.ix[train_dataset.MasVnrType.isnull(), 'MasVnrType'] = 'None'
train_dataset.ix[train_dataset.MasVnrType == 'None', 'MasVnrArea'] = 0

# Fix FireplaceQu - set to NA only if the property has no fireplace. the number of fireplaces matches the number of null FireplaceQu
train_dataset.groupby('Fireplaces').Id.nunique()
train_dataset.ix[train_dataset.Fireplaces == 0, 'FireplaceQu'] = 'NA'

# Fix GarageType, GarageYrBlt, GarageFinish, GarageQual, GarageCond -- set to NA if no garage
train_dataset.groupby('GarageCars').Id.nunique()
train_dataset.ix[train_dataset.GarageCars.isnull(), 'GarageCars'] = 0
train_dataset.ix[train_dataset.GarageQual.isnull(), 'GarageQual'] = 'TA'
train_dataset.ix[train_dataset.GarageCars == 0, 'GarageType'] = 'NA'
train_dataset.ix[train_dataset.GarageType == 'NA', ['GarageYrBlt', 'GarageArea']] = 0
train_dataset.ix[train_dataset.GarageType == 'NA', ['GarageFinish','GarageQual','GarageCond']] = 'NA'
train_dataset.ix[train_dataset.GarageCars == 0, 'GarageQual'] = 'NA'
train_dataset.ix[train_dataset.GarageCars == 0, 'GarageCond'] = 'NA'

train_dataset.groupby('GarageQual').Id.nunique()
train_dataset.ix[train_dataset.Id == 2127, 'GarageYrBlt'] = 0
train_dataset.ix[train_dataset.Id == 2127, 'GarageFinish'] = 'Fin'
train_dataset.ix[train_dataset.Id == 2127, 'GarageQual'] = 'TA'
train_dataset.ix[train_dataset.Id == 2127, 'GarageCond'] = 'TA'


# Fix PoolQC - set to NA if no pool
train_dataset.groupby('PoolArea').Id.nunique()
train_dataset.ix[train_dataset.PoolArea == 0, 'PoolQC'] = 'NA'
train_dataset.ix[train_dataset.PoolArea != 0, 'PoolQC'] = 'TA'

# Fix Fence 
train_dataset.groupby('Fence').Id.nunique()
train_dataset.ix[train_dataset.Fence.isnull(), 'Fence'] = 'NA'

# Fix MiscFeature 
train_dataset.groupby('MiscFeature').Id.nunique()
train_dataset.ix[train_dataset.MiscFeature.isnull(), 'MiscFeature'] = 'NA'

# Fix BsmtQual
train_dataset.ix[train_dataset.BsmtQual.isnull(), ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', ]] = 'NA'
train_dataset.ix[train_dataset.BsmtQual == 'NA', ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']] = 0
train_dataset.groupby('BsmtCond').Id.nunique()
train_dataset.ix[train_dataset.BsmtCond.isnull(), 'BsmtCond'] = 'TA'
train_dataset = train_dataset.drop('BsmtExposure',1)
train_dataset = train_dataset.drop('BsmtFinType2',1)
train_dataset = train_dataset.drop('Electrical',1)
train_dataset = train_dataset.drop('Utilities',1) #all rows populated with AllPub. Adds no differentiating value
 
# Fix MSZoning
train_dataset.groupby('MSZoning').Id.nunique()
train_dataset.ix[train_dataset.MSZoning.isnull(), 'MSZoning'] = 'RL'

# Fix Exterior1st
train_dataset.groupby('Exterior1st').Id.nunique()
train_dataset.ix[train_dataset.Exterior1st.isnull(), ['Exterior1st','Exterior2nd']] = 'VinylSd'

# Fix Functional
train_dataset.ix[train_dataset.Functional.isnull(), ['Functional']] = 'Typ'

# Fix KitchenQual
train_dataset.ix[train_dataset.KitchenQual.isnull(), ['KitchenQual']] = 'TA'

# Fix SaleType
train_dataset.ix[train_dataset.SaleType.isnull(), 'SaleType'] = 'WD'

train_dataset.info()

#temp_df = train_dataset
#train_dataset.groupby('SaleType').Id.nunique()
#temp_df = temp_df[temp_df['GarageQual'].isnull()]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 14:04:40 2016

@author: davideng
"""

################ Clean test dataset

test_dataset.isnull().sum()
# Fix Alley -- Does property have alley: yes or no
test_dataset.groupby('Alley').Id.nunique()
test_dataset.ix[test_dataset.Alley.notnull(), 'Alley'] = 1
test_dataset.ix[test_dataset.Alley.isnull(), 'Alley'] = 0

# Fix LotFrontage -- change to median lot size
imputer2 = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer2 = imputer2.fit(test_dataset[['LotFrontage']])
test_dataset[['LotFrontage']] = imputer.transform(test_dataset[['LotFrontage']])

# Fix MasVnrType -- updated the 8 null fields with 'None' since 'None' is majority of dataset 
# Fix MasVnrArea -- updated with 0 since 'MasVnrType' set to 'None'
test_dataset.groupby('MasVnrType').Id.nunique()
test_dataset.ix[test_dataset.MasVnrType.isnull(), 'MasVnrType'] = 'None'
test_dataset.ix[test_dataset.MasVnrType == 'None', 'MasVnrArea'] = 0

# Fix FireplaceQu - set to NA only if the property has no fireplace. the number of fireplaces matches the number of null FireplaceQu
test_dataset.groupby('Fireplaces').Id.nunique()
test_dataset.ix[test_dataset.Fireplaces == 0, 'FireplaceQu'] = 'NA'

# Fix GarageType, GarageYrBlt, GarageFinish, GarageQual, GarageCond -- set to NA if no garage
test_dataset.groupby('GarageCars').Id.nunique()
test_dataset.ix[test_dataset.GarageCars.isnull(), 'GarageCars'] = 0
test_dataset.ix[test_dataset.GarageQual.isnull(), 'GarageQual'] = 'TA'
test_dataset.ix[test_dataset.GarageCars == 0, 'GarageType'] = 'NA'
test_dataset.ix[test_dataset.GarageType == 'NA', ['GarageYrBlt', 'GarageArea']] = 0
test_dataset.ix[test_dataset.GarageType == 'NA', ['GarageFinish','GarageQual','GarageCond']] = 'NA'
test_dataset.ix[test_dataset.GarageCars == 0, 'GarageQual'] = 'NA'
test_dataset.ix[test_dataset.GarageCars == 0, 'GarageCond'] = 'NA'

test_dataset.groupby('GarageQual').Id.nunique()
test_dataset.ix[test_dataset.Id == 2127, 'GarageYrBlt'] = 0
test_dataset.ix[test_dataset.Id == 2127, 'GarageFinish'] = 'Fin'
test_dataset.ix[test_dataset.Id == 2127, 'GarageQual'] = 'TA'
test_dataset.ix[test_dataset.Id == 2127, 'GarageCond'] = 'TA'


# Fix PoolQC - set to NA if no pool
test_dataset.groupby('PoolArea').Id.nunique()
test_dataset.ix[test_dataset.PoolArea == 0, 'PoolQC'] = 'NA'
test_dataset.ix[test_dataset.PoolArea != 0, 'PoolQC'] = 'TA'

# Fix Fence 
test_dataset.groupby('Fence').Id.nunique()
test_dataset.ix[test_dataset.Fence.isnull(), 'Fence'] = 'NA'

# Fix MiscFeature 
test_dataset.groupby('MiscFeature').Id.nunique()
test_dataset.ix[test_dataset.MiscFeature.isnull(), 'MiscFeature'] = 'NA'

# Fix BsmtQual
test_dataset.ix[test_dataset.BsmtQual.isnull(), ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', ]] = 'NA'
test_dataset.ix[test_dataset.BsmtQual == 'NA', ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']] = 0
test_dataset.groupby('BsmtCond').Id.nunique()
test_dataset.ix[test_dataset.BsmtCond.isnull(), 'BsmtCond'] = 'TA'
test_dataset = test_dataset.drop('BsmtExposure',1)
test_dataset = test_dataset.drop('BsmtFinType2',1)
test_dataset = test_dataset.drop('Electrical',1)
test_dataset = test_dataset.drop('Utilities',1) #all rows populated with AllPub. Adds no differentiating value
 
# Fix MSZoning
test_dataset.groupby('MSZoning').Id.nunique()
test_dataset.ix[test_dataset.MSZoning.isnull(), 'MSZoning'] = 'RL'

# Fix Exterior1st
test_dataset.groupby('Exterior1st').Id.nunique()
test_dataset.ix[test_dataset.Exterior1st.isnull(), ['Exterior1st','Exterior2nd']] = 'VinylSd'

# Fix Functional
test_dataset.ix[test_dataset.Functional.isnull(), ['Functional']] = 'Typ'

# Fix KitchenQual
test_dataset.ix[test_dataset.KitchenQual.isnull(), ['KitchenQual']] = 'TA'

# Fix SaleType
test_dataset.ix[test_dataset.SaleType.isnull(), 'SaleType'] = 'WD'

test_dataset.info()

#temp_df = test_dataset
#test_dataset.groupby('SaleType').Id.nunique()
#temp_df = temp_df[temp_df['GarageQual'].isnull()]

################ Encode categorical data
# Step 2: Check categorial columns and change to numeric
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

for i in train_dataset.columns:
    if train_dataset[i].dtypes == np.object:
        varName = 'labelencoder' + i
        varName = LabelEncoder()
        train_dataset[i] = varName.fit_transform(train_dataset[i])
        
for j in test_dataset.columns:
    if test_dataset[j].dtypes == np.object:
        varName2 = 'labelencoder' + j + j
        varName2 = LabelEncoder()
        test_dataset[j] = varName2.fit_transform(test_dataset[j])

#onehotencoder = OneHotEncoder(categorical_features = [3])
#X = onehotencoder.fit_transform(X).toarray()

# Step 3: Create train and test datasets
X_train = train_dataset.iloc[:, :-1].values
y_train = train_dataset.iloc[:, 76].values
X_test = test_dataset.iloc[:,:].values

#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Step 4: Perform Random Forest
# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 5000, random_state = 0)
regressor.fit(X_train, y_train)
regressor.score(X_train, y_train)

# Step 5: Predicting test results
y_pred = regressor.predict(X_test)
submission = pd.DataFrame({
            'Id': test_dataset['Id'],
            'SalePrice': y_pred
            })
submission.to_csv('salePrice.csv', index=False)

## Step 6: Visualising the Random Forest Regression results (higher resolution)
#X_grid = np.arange(min(X_train[:,0]), max(X_train[:,0]), 0.01)
#X_grid = X_grid.reshape((len(X_grid), 1))
#plt.scatter(X_train[:,0], y_train[:], color = 'red')
#plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
#plt.title('Titanic (Random Forest Regression)')
#plt.xlabel('Position level')
#plt.ylabel('Salary')
#plt.show()


