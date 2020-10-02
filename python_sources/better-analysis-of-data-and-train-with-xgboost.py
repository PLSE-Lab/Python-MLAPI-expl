#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/train.csv')


# In[ ]:


train.head()


# In[ ]:


train_label = train['SalePrice']


# In[ ]:


numeric_columns = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 
                  'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',  
                  '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'FullBath',
                  'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',
                   'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',  '3SsnPorch', 'ScreenPorch', 
                   'PoolArea', 'MiscVal', 'MoSold', 'YrSold']


# # Checking Categorical data

# In[ ]:


index = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 
        'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
        'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond',
        'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'Heating', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType',
        'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive','PoolQC', 'Fence', 'MiscFeature',
        'SaleType', 'SaleCondition']


# In[ ]:


i = 0
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## Five types for MSZoning: [C, FV, RH, RL, RM]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## Two types for Street : [Grvl, Pave]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## Two types for Alley : [Grvl, Pave]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## Four types for LotShape: [IR1, IR2, IR3, Reg]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


#  ## Four types for LandContour: [Bnk, HLS, Low, Lvl]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## Two types for Utilities: [AllPub, NoSeWa]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## Five types for LotConfig: [Corner, CulDSac, FR2, FR3, Inside]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## Three types for LandSlope: [Gtl, Mod, Sev]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## 25 types for Neighborhood: [Blmngtn, Blueste, BrDale, BrkSide, ..., Somerst, StoneBr, Timber, Veenker]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## 9 types for Condition1: [Artery, Feedr, Norm, PosA, ..., RRAe, RRAn, RRNe, RRNn]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## 8 types for Condition2: [Artery, Feedr, Norm, PosA, PosN, RRAe, RRAn, RRNn]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## 5 types for BldgType: [1Fam, 2fmCon, Duplex, Twnhs, TwnhsE]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## 8 types for HouseStyle: [1.5Fin, 1.5Unf, 1Story, 2.5Fin, 2.5Unf, 2Story, SFoyer, SLvl]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## 6 types for RoofStyle: [Flat, Gable, Gambrel, Hip, Mansard, Shed]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## 8 types for RoofMatl: [ClyTile, CompShg, Membran, Metal, Roll, Tar&Grv, WdShake, WdShngl]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## 15 types for Exterior1st: [AsbShng, AsphShn, BrkComm, BrkFace, ..., Stucco, VinylSd, Wd Sdng, WdShing]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## 16 types for Exterior2nd: [AsbShng, AsphShn, Brk Cmn, BrkFace, ..., Stucco, VinylSd, Wd Sdng, Wd Shng]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## 4 types for MasVnrType: [BrkCmn, BrkFace, None, Stone]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## 4 types for ExterQual: [Ex, Fa, Gd, TA]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## 5 types for ExterCond: [Ex, Fa, Gd, Po, TA]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## 6 types for Foundation: [BrkTil, CBlock, PConc, Slab, Stone, Wood]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## 4 types for BsmtQual: [Ex, Fa, Gd, TA]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## 4 types for BsmtCond: [Fa, Gd, Po, TA]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## 4 types for BsmtExposure: [Av, Gd, Mn, No]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## 6 types for BsmtFinType1: [ALQ, BLQ, GLQ, LwQ, Rec, Unf]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## 6 types for BsmtFinType2: [ALQ, BLQ, GLQ, LwQ, Rec, Unf]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## 6 types for Heating: [Floor, GasA, GasW, Grav, OthW, Wall]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## 2 types for CentralAir: [N, Y]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## 5 types for Electrical: [FuseA, FuseF, FuseP, Mix, SBrkr]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## 4 types for KitchenQual: [Ex, Fa, Gd, TA]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## 7 types for Functional: [Maj1, Maj2, Min1, Min2, Mod, Sev, Typ]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## 5 types for FireplaceQu: [Ex, Fa, Gd, Po, TA]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## 6 types for GarageType: [2Types, Attchd, Basment, BuiltIn, CarPort, Detchd]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## 3 types for GarageFinish: [Fin, RFn, Unf]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## 5 types for GarageQual: [Ex, Fa, Gd, Po, TA]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## 5 types for GarageCond: [Ex, Fa, Gd, Po, TA]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## 3 types for PavedDrive: [N, P, Y]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## 3 types for PoolQC: [Ex, Fa, Gd]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## 4 types for Fence: [GdPrv, GdWo, MnPrv, MnWw]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## 4 types for MiscFeature: [Gar2, Othr, Shed, TenC]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# ## 9 types for SaleType: [COD, CWD, Con, ConLD, ..., ConLw, New, Oth, WD]

# In[ ]:


df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']


# In[ ]:


## 6 types for SalesCondition: [Abnorml, AdjLand, Alloca, Family, Normal, Partial]


# In[ ]:


num = train[numeric_columns]


# In[ ]:


index = train[index]


# In[ ]:


from sklearn.model_selection import train_test_split
# Train_test split with 25% test size
train_data, test_data, train_labels, test_labels = train_test_split(num, 
                                                                    train_label, 
                                                                    test_size=0.20)


# In[ ]:


import xgboost as xgb
import numpy as np
# Flatten columns
train_labels = np.ravel(train_labels)
test_labels = np.ravel(test_labels)

# Create DMatrix for xgboost
D_train = xgb.DMatrix(data=train_data, silent=1, nthread=-1, label =train_labels)
D_test  = xgb.DMatrix(data=test_data,  silent=1, nthread=-1, label =test_labels)


# In[ ]:


param = {'silent' : 1,
         'learning_rate' : 0.03,
         'max_depth': 10,
         'tree_method': 'exact',
         'objective': 'reg:linear'
         }

n_rounds = 300

watch_list = [(D_train, 'train'), (D_test, 'eval')]
bst = xgb.train(param, D_train, n_rounds, watch_list, early_stopping_rounds = 15)
pred = bst.predict( D_test )
predictions = [np.around(value) for value in pred]


# In[ ]:


from sklearn.metrics import accuracy_score, precision_score
accuracy = accuracy_score(test_labels, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:


df2 = test_data
df2.loc[:,('pred')] = list(predictions)
df2.loc[:,('real')] = list(test_labels)
df2.groupby("pred").agg("count")


# In[ ]:


df2.groupby("real").agg("count")


# In[ ]:


xgb.plot_importance(bst)


# In[ ]:


# Using model to test data
test_df = pd.read_csv('../input/test.csv')
test_df


# In[ ]:


num = test_df[numeric_columns]
D_test  = xgb.DMatrix(data=num,  silent=1, nthread=-1)
pred = bst.predict( D_test )
predictions = [np.around(value) for value in pred]


# In[ ]:


test_df['SalePrice'] = predictions
test_df


# In[ ]:


submission = test_df[['Id', 'SalePrice']]


# In[ ]:


submission.to_csv('submission.csv', index = False)

