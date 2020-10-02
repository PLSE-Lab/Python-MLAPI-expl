# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))

# Read DataFrame
train = pd.read_csv('../input/train.csv')

# Create target: SalePrice
Y = train['SalePrice']

# Create variables excluding Id and SalePrice
X = train.iloc[:, 1:-1]

cols = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley',
       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
       'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',
       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
       'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',
       'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
       'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt',
       'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
       'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
       'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal',
       'MoSold', 'YrSold', 'SaleType', 'SaleCondition']

quality_map = {'Ex': 5,
               'Gd': 4,
               'TA': 3,
               'Fa': 2,
               'Po': 1,
               'NA': 0}

for i in ['ExterCond',
          'ExterQual',
          'ExterCond',
          'ExterQual',
          'BsmtQual',
          'BsmtCond',
          'HeatingQC',
          'KitchenQual',
          'FireplaceQu',
          'GarageCond',
          'GarageQual',
          'PoolQC'
          ]:
    X[i] = X[i].map(quality_map)


# First test: small number of variables
sub = ['Neighborhood',
       'OverallQual',
       'OverallCond',
       'YearBuilt',
       'YearRemodAdd',
       '1stFlrSF',
       '2ndFlrSF',
       'GarageArea',
       'MoSold',
       'YrSold',
       'PoolArea',
       'ExterCond',
       'ExterQual',
       'ExterCond',
       'ExterQual',
       'BsmtQual',
       'BsmtCond',
       'HeatingQC',
       'KitchenQual',
       'FireplaceQu',
       'GarageCond',
       'GarageQual',
       'PoolQC'
       ]

X = X[sub]

# Add Neighborhood dummies
X = pd.concat([X, pd.get_dummies(X['Neighborhood'])], axis=1)
X.drop('Neighborhood', axis=1, inplace=True)

X = X.fillna(0)
X.to_csv('X', index=False)
Y.to_csv('Y', index=False, header='SalesPrice')