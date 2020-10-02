# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

train = pd.read_csv("../input/train.csv")

Features = ["SalePrice","MSSubClass","MSZoning","LotArea","LotShape","LandContour","Neighborhood",
                    "BldgType","HouseStyle","OverallQual","OverallCond","YearBuilt",
                    "YearRemodAdd","Exterior1st","Exterior2nd","MasVnrArea","ExterQual",
                    "Foundation","BsmtQual","BsmtCond","BsmtFinType1","BsmtFinSF1",
                    "BsmtFinType2","BsmtUnfSF","TotalBsmtSF","HeatingQC","CentralAir",
                    "1stFlrSF","2ndFlrSF","GrLivArea","BsmtFullBath","FullBath","HalfBath",
                    "BedroomAbvGr","KitchenAbvGr","KitchenQual","TotRmsAbvGrd","Functional",
                    "Fireplaces","FireplaceQu","GarageType","GarageYrBlt","GarageFinish",
                    "GarageCars","GarageArea","GarageQual","GarageCond","PavedDrive","WoodDeckSF",
                    "OpenPorchSF","Fence"]

train_df = train[ Features ]

train_df.head()

quantitativeFeatures = [f for f in train_df.columns if train_df.dtypes[f] != 'object']

qualitativeFeatures = [f for f in train_df.columns if train_df.dtypes[f] == 'object']

import matplotlib.pyplot as plt
plt.hist(train_df["SalePrice"])
plt.show()

train_df["SalePrice"] = np.log1p(train_df["SalePrice"])
plt.hist(train_df["SalePrice"])
plt.show()

train_df.update(train_df[qualitativeFeatures].fillna('None'))
train_df.update(train_df[quantitativeFeatures].fillna(-1))
train_df.isnull().sum()

y = train_df.SalePrice.reset_index(drop=True)
train_df = train_df.drop(['SalePrice'], axis=1)
for c in qualitativeFeatures:
    train_df[c] =  pd.Categorical(train_df[c])
    train_df[c] = train_df[c].cat.codes

from sklearn.preprocessing import *
#scaler = StandardScaler()
scaler = MinMaxScaler()
scaler.fit(train_df)
train_df = scaler.transform(train_df)

from xgboost.sklearn import XGBRegressor
#classifier = XGBRegressor(learning_rate=0.01,n_estimators=3000,
#                                    max_depth=2, min_child_weight=0,
#                                     gamma=0, subsample=0.7,
#                                     colsample_bytree=0.7,
#                                     objective='reg:squarederror', nthread=-1,
#                                     scale_pos_weight=1, seed=27,
#                                     reg_alpha=0.0006)

classifier = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

#from lightgbm import LGBMRegressor
#classifier = LGBMRegressor(objective='regression', num_leaves=4, learning_rate=0.01,
#                                    n_estimators=5000, max_bin=200,
#                                    bagging_fraction=0.75, bagging_freq=5,
#                                    bagging_seed=7,
#                                    feature_fraction=0.2, feature_fraction_seed=7, verbose=-1)

#from sklearn.ensemble import GradientBoostingRegressor
#classifier = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.01,
#                                   max_depth=5, max_features='sqrt',
#                                   min_samples_leaf=20, min_samples_split=15, 
#                                   loss='huber', random_state =42)



classifier.fit(train_df, y)

TestFeatures = ["MSSubClass","MSZoning","LotArea","LotShape","LandContour","Neighborhood",
                    "BldgType","HouseStyle","OverallQual","OverallCond","YearBuilt",
                    "YearRemodAdd","Exterior1st","Exterior2nd","MasVnrArea","ExterQual",
                    "Foundation","BsmtQual","BsmtCond","BsmtFinType1","BsmtFinSF1",
                    "BsmtFinType2","BsmtUnfSF","TotalBsmtSF","HeatingQC","CentralAir",
                    "1stFlrSF","2ndFlrSF","GrLivArea","BsmtFullBath","FullBath","HalfBath",
                    "BedroomAbvGr","KitchenAbvGr","KitchenQual","TotRmsAbvGrd","Functional",
                    "Fireplaces","FireplaceQu","GarageType","GarageYrBlt","GarageFinish",
                    "GarageCars","GarageArea","GarageQual","GarageCond","PavedDrive","WoodDeckSF",
                    "OpenPorchSF","Fence"]

test = pd.read_csv("../input/test.csv")
test_df = test[ TestFeatures ]

quantitativeFeaturesTest = [f for f in test_df.columns if test_df.dtypes[f] != 'object']

qualitativeFeaturesTest = [f for f in test_df.columns if test_df.dtypes[f] == 'object']

test_df.update(test_df[qualitativeFeaturesTest].fillna('None'))
test_df.update(test_df[quantitativeFeaturesTest].fillna(-1))
for c in qualitativeFeatures:
    test_df[c] =  pd.Categorical(test_df[c])
    test_df[c] = test_df[c].cat.codes

test_df = scaler.transform(test_df)

pred_Y = classifier.predict(test_df)
pred_Y = np.expm1(pred_Y)

f = open("housing_xgr4.csv","w")
f.write("Id,SalePrice\n")
for i in range(0,len(test_df)):
    f.write("{},{}\n".format(test["Id"][i],pred_Y[i]))
f.close()