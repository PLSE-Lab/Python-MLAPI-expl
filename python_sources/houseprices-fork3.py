#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# In this fork we are testing and learning new machine learning models. 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import norm, skew
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import Imputer
# from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.linear_model import RidgeCV


# In[38]:


#     train = pd.read_csv("train.csv")
#     test = pd.read_csv("test.csv")
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

test_id = test.Id
# Set target variable
y_train = np.log1p(train['SalePrice'])
y = train['SalePrice']
# concatinate test and train data sets before making changes
features = pd.concat([train,test], keys=['train','test'],names=['train','test'])
features = features.drop(['Id'], axis=1)
# feature engineering
features['TotalSqFt'] = features['1stFlrSF'] + features['2ndFlrSF'] + features['TotalBsmtSF']
####################
# create formula to view all empty columns 
all_cols = features.columns
cols_nan = []
cols_missing_total = []
data_type = []
unique_val = []
for col in all_cols:
    if features[col].isnull().sum() > 0:
        cols_nan.append(col)
        cols_missing_total.append(features[col].isnull().sum())
        data_type.append(features[col].dtype)
        unique_val.append(features[col].unique())
cols_nan = pd.Series(data=cols_nan)
cols_missing_total = pd.Series(data=cols_missing_total)
data_type = pd.Series(data=data_type)
unique_val = pd.Series(data=unique_val)
df_missing_values = pd.concat([cols_nan,cols_missing_total,data_type, unique_val], 
                               keys=['cols_nan','cols_missing_total','data_type','unique_val'],
                               axis=1)
df_missing_values.sort_values('cols_missing_total',ascending=False);
####################
features[features['GarageCars'].isnull()]
features['GarageArea'].mode()[0];
features['Electrical'] = features['Electrical'].fillna('SBrkr')
features['Functional'] = features['Functional'].fillna('Typ')    
features['SaleType'] = features['SaleType'].fillna('WD')
features['TotalSqFt'] = features['TotalSqFt'].fillna(features['TotalSqFt'].mean())
features['KitchenQual'] = features['KitchenQual'].fillna('TA')
features['Exterior1st'] = features['Exterior1st'].fillna('VinylSd')
features['Exterior2nd'] = features['Exterior2nd'].fillna('VinylSd')
features['Utilities'] = features['Utilities'].fillna('AllPub')
features['MSZoning'] = features['MSZoning'].fillna('RL')
####################
missing_values = []
for col in ('TotalBsmtSF', 'SaleType', 'KitchenQual', 'BsmtFinSF1', 'GarageArea', 'BsmtFinSF2', 'Exterior2nd', 'Exterior1st', 'Electrical', 'BsmtUnfSF', 'TotalSqFt'):
    missing_values.append(features[features[col].isnull()])
df_missing_val = pd.concat(missing_values)
# df_missing_val
features['PoolQC'] = features['PoolQC'].fillna('None')
features['MiscFeature'] = features['MiscFeature'].fillna('None')
features['Alley'] = features['Alley'].fillna('None')
features['Fence'] = features['Fence'].fillna('None')
features['FireplaceQu'] = features['FireplaceQu'].fillna('None')
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtHalfBath', 'BsmtFullBath'):
    features[col] = features[col].fillna(0)
for col in ('BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1'):
    features[col] = features[col].fillna('None')
for col in ('GarageFinish','GarageCond', 'GarageQual', 'GarageType'):
    features[col] = features[col].fillna('None')
for col in ('GarageArea', 'GarageCars', 'GarageYrBlt'):
    features[col] = features[col].fillna(0)
features['MasVnrType'] = features['MasVnrType'].fillna('None')
features['MasVnrArea'] = features['MasVnrArea'].fillna(0)
features['LotFrontage'] = features['LotFrontage'].fillna(features['LotFrontage'].median())
# CONVERTING NUMERICALS TO STRING TYPES, WHERE IT MAKES SENSE
# convert numerical columns to string types, for which should be string.
# create list of all numerical columns 
num_cols = features.dtypes[features.dtypes != 'object'].index

# Outlier corrections
features.loc[('test',1132),'GarageYrBlt'] = 2007
# Converting to string type
features['MoSold'] = features['MoSold'].astype(str) # MoSold finalized
features['MSSubClass'] = features['MSSubClass'].astype(str) # MSSubClass finalized
######################
# # manual label encoding
KitchenQual_map = {'Gd':3, 'TA':2, 'Ex':4, 'Fa':1}
features['KitchenQual'] = features['KitchenQual'].map(KitchenQual_map)

GarageQual_map = {'TA':2, 'Fa':3, 'Gd':4, 'None':0, 'Ex':5, 'Po':1}
features['GarageQual'] = features['GarageQual'].map(GarageQual_map)

ExterCond_map = {'TA':3, 'Gd':4, 'Fa':2, 'Po':1, 'Ex':5}
features['ExterCond'] = features['ExterCond'].map(ExterCond_map)

ExterQual_map = {'Gd':3, 'TA':2, 'Ex':4, 'Fa':1}
features['ExterQual'] = features['ExterQual'].map(ExterQual_map)

FireplaceQu_map = {'None':0, 'TA':3, 'Gd':4, 'Fa':2, 'Ex':5, 'Po':1}
features['FireplaceQu'] = features['FireplaceQu'].map(FireplaceQu_map)

CentralAir_map = {'Y':1, 'N':0}
features['CentralAir'] = features['CentralAir'].map(CentralAir_map)

GarageCond_map = {'TA':3, 'Fa':2, 'None':0, 'Gd':4, 'Po':1, 'Ex':5}
features['GarageCond'] = features['GarageCond'].map(GarageCond_map)

HeatingQC_map = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1}
features['HeatingQC'] = features['HeatingQC'].map(HeatingQC_map)

Street_map = {'Pave':1, 'Grvl':0}
features['Street'] = features['Street'].map(Street_map)

BsmtCond_map = {'TA':3, 'Gd':4, 'None':0, 'Fa':2, 'Po':1}
features['BsmtCond'] = features['BsmtCond'].map(BsmtCond_map)

BsmtExposure_map = {'No':1, 'Gd':4, 'Mn':2, 'Av':3, 'None':0}
features['BsmtExposure'] = features['BsmtExposure'].map(BsmtExposure_map)

BsmtQual_map = {'Gd':3, 'TA':2, 'Ex':4, 'None':0, 'Fa':1}
features['BsmtQual'] = features['BsmtQual'].map(BsmtQual_map)

HouseStyle_map = {'2Story':7, '1Story':5, '1.5Fin':4, '1.5Unf':1, 'SFoyer':2, 'SLvl':3, '2.5Unf':6, '2.5Fin':8}
features['HouseStyle'] = features['HouseStyle'].map(HouseStyle_map)

LotShape_map = {'Reg':4, 'IR1':3, 'IR2':2, 'IR3':1}
features['LotShape'] = features['LotShape'].map(LotShape_map)

LandSlope_map = {'Gtl':3, 'Mod':2, 'Sev':1}
features['LandSlope'] = features['LandSlope'].map(LandSlope_map)

MSSubClass_map = {'60':15, '20':12, '70':11, '50':8, '190':9, '45':2, '90':10, '120':13, '30':3, '85':4, '80':7, '160':5, '75':14, '180':1, '40':6, '150':0}
features['MSSubClass'] = features['MSSubClass'].map(MSSubClass_map)

BldgType_map = {'1Fam':4, '2fmCon':1, 'Duplex':3, 'TwnhsE':5, 'Twnhs':2}
features['BldgType'] = features['BldgType'].map(BldgType_map)

BsmtFinType1_map = {'GLQ':6, 'ALQ':5, 'Unf':1, 'Rec':3, 'BLQ':4, 'None':0, 'LwQ':2}
features['BsmtFinType1'] = features['BsmtFinType1'].map(BsmtFinType1_map)

BsmtFinType2_map = {'GLQ':6, 'ALQ':5, 'Unf':1, 'Rec':3, 'BLQ':4, 'None':0, 'LwQ':2}
features['BsmtFinType2'] = features['BsmtFinType2'].map(BsmtFinType2_map)

Fence_map = {'None':0, 'MnPrv':1, 'GdWo':2, 'GdPrv':2, 'MnWw':1}
features['Fence'] = features['Fence'].map(Fence_map)

Functional_map = {'Typ':7, 'Min1':6, 'Maj1':3, 'Min2':5, 'Mod':4, 'Maj2':2, 'Sev':1}
features['Functional'] = features['Functional'].map(Functional_map)

MiscFeature_map = {'None':0, 'Shed':1, 'Gar2':2, 'Othr':1, 'TenC':3}
features['MiscFeature'] = features['MiscFeature'].map(MiscFeature_map)

# PoolQC_map = {'None':0, 'Ex':3, 'Fa':1, 'Gd':2}
# features['PoolQC'] = features['PoolQC'].map(PoolQC_map)

Utilities_map = {'AllPub':2, 'NoSeWa':1}
features['Utilities'] = features['Utilities'].map(Utilities_map)

Foundation_map = {'PConc':5, 'CBlock':3, 'BrkTil':0, 'Wood':2, 'Slab':4, 'Stone':1}
features['Foundation'] = features['Foundation'].map(Foundation_map)
# PCond better than CBlock. Stone is the worst. 

Heating_map = {'GasA':5, 'GasW':6, 'Grav':2, 'Wall':1, 'OthW':4, 'Floor':3}
features['Heating'] = features['Heating'].map(Heating_map)
###########
# Remove feature engineering until we finish all object arrays
features['AgeYears'] = features['YearBuilt'].astype('int64') # finalized
features['AgeYears'] = 2019 - features['AgeYears']
features['YearRemodAdd'] = features['YearRemodAdd'].astype('int64')
features['YearRemodAdd'] = 2019 - features['YearRemodAdd'] # finalized
features['YrSold'] = features['YrSold'].astype('str') # finalized, 5 categories

#### working on GarageYrBlt 
# testing GarageYrBlt as age
features['GarageYrBlt'] = features['GarageYrBlt'].astype(float)
features['GarageYrBlt'] = features['GarageYrBlt'].astype(int)
features['GarageYrBlt'] = 2019 - features['GarageYrBlt']
# features['GarageYrBlt'].unique()

# FEATURE ENGINEERING
features['fe_OverallQual'] = features['OverallQual'] ** 5
features['fe_TotalSqFt'] = features['TotalSqFt'] ** 2
features['PorchTotalSF'] = features['3SsnPorch'] + features['EnclosedPorch'] + features['OpenPorchSF'] + features['ScreenPorch']

# Adding these features hurt our score. 
# features['fe_BsmtFinSFTotal'] = features['BsmtFinSF1'] + features['BsmtFinSF2']
# features['fe_BsmtTotalBath'] = features['BsmtFullBath'] + features['BsmtHalfBath']
# features['fe_TotalBath'] = features['FullBath'] + features['HalfBath'] + features['BsmtFullBath'] + features['BsmtHalfBath']

# Dropping features 
# features = features.drop(['Condition2','LotFrontage','MoSold','OpenPorchSF','Alley','GarageType',
#                           'GarageArea','Exterior2nd','SaleType','GarageYrBlt','Heating','Fence',
#                           'PavedDrive','GarageFinish', 'MasVnrType','YrSold','Electrical','LotShape',
#                           'ExterQual','MasVnrArea','RoofStyle','3SsnPorch','Street','Fireplaces',
#                           'ExterCond','MiscFeature','BsmtFinType2','BldgType','PoolArea','Utilities',
#                           'MiscVal','PoolQC'],axis=1)
features = features.drop(['PoolArea','PoolQC','Condition2','LotFrontage','MoSold','OpenPorchSF','Alley','GarageType',
                          'GarageArea','Exterior2nd','SaleType'],axis=1)


# In[39]:


#Validation function
def rmsle_cv(model):
    kfolds = KFold(n_splits=10, shuffle=True,random_state=23)
    rmse = np.sqrt(-cross_val_score(model, train, y_train.values, scoring="neg_mean_squared_error", cv = kfolds))
    return(rmse)


# In[40]:


# get_dummies
# continue, look at outliers and then start adding more algorithms, start working for j book.
features = pd.get_dummies(features)

# Didn't help our score, by fixing skewed features. 
# features['TotalSqFt'] = np.cbrt(features['TotalSqFt'])
# features['fe_TotalSqFt'] = np.log1p(features['fe_TotalSqFt']) # finalized transformation
# features['GrLivArea'] = np.log(features['GrLivArea']) # finalized transformation
# features['1stFlrSF'] = np.log(features['1stFlrSF']) # finalized transformation
# features['MasVnrArea'] = np.log1p(features['MasVnrArea']) # finalized transformation
# features['fe_BsmtFinSFTotal'] = np.sqrt(features['fe_BsmtFinSFTotal']) # finalized transformation
# features['MiscVal'] = np.log1p(features['MiscVal']) # finalized transformation

# splitting train and test
train = features.loc['train']
test = features.loc['test']


# In[41]:


train = train[train['GrLivArea'] < 4500] # finalized outliers
# target
y = train['SalePrice'].reset_index(drop=True)
y = np.log1p(y)
# outliers
outliers = [30, 88, 462, 631, 1322]
train = train.drop(train.index[outliers])
y = y.drop(y.index[outliers])
# split train and test
train = train.drop('SalePrice',axis=1)
test = test.drop('SalePrice',axis=1)


# In[42]:


# ElasticNetCV algorithm
from sklearn.linear_model import ElasticNetCV
l1_ratio = [.1, .5, .7, .9, .95, .99, 1]
e_alphas = [.0001,.0002,.00021,.00022,.00023,.00024,.00025,.00026,.00027,.00028,.00029]

model_encv_pipeline  = make_pipeline(
    RobustScaler(),
    ElasticNetCV(
                 l1_ratio = l1_ratio,
                 alphas=e_alphas,
                 max_iter=1e7,
                 random_state=42
                )
    )
model_encv_pipeline.fit(train,y)
results_encv = model_encv_pipeline.predict(test)
preds_encv = np.expm1(results_encv)
preds_encv = pd.Series(preds_encv)
preds_encv
# Calculate scoring
# score = rmsle_cv(model_encv_pipeline)
# print('model_encv_pipeline')
# print(score.mean())
# 0.12988184156085442 base
# 0.11911870480203288 mms
# 0.11754055682914899 w/o mms
# 0.12635 kaggle score w/o mms
# 0.13342235811169217 w/o rs
# 0.1190313796538196 w/ rs & mms
# 0.1175402992637199 w/ rs w/o mms
# 0.11053887975460697 w/rs w/o mms w/ dropping outliers
# 0.11053617268832287 w/rs w/ mms w/ dropping outliers
# 0.1105387940505933 w/rs w/ mms w/ dropping outliers w/dropping features
# 0       115214.330946


# In[ ]:


# Algorithm - LassoCV
from sklearn.linear_model import LassoCV
alpha_lasso = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10]
model_lcv = make_pipeline(
                           MinMaxScaler(),
                           RobustScaler(),
                           LassoCV(
                                   alphas=alpha_lasso,
                                   random_state=42,
                                   cv=3
                                   ))
model_lcv.fit(train,y)
results_lcv = model_lcv.predict(test)
preds_lcv = np.expm1(results_lcv)
preds_lcv = pd.Series(preds_lcv)
# preds_lcv
# 0       113937.504971


# In[ ]:


# RidgeCV model
from sklearn.linear_model import RidgeCV
# Algorithm - RidgeCV
alpha_rcv = [1e-2, 1e-1, .2, .3, .4, .5, .6, .7, .9, 1]
model_rcv = make_pipeline(
                          MinMaxScaler(),
                          RobustScaler(),
                          RidgeCV(
                                  alphas=alpha_rcv
#                                   cv=3
                                   ))
model_rcv.fit(train,y)
results_rcv = model_rcv.predict(test)
preds_rcv = np.expm1(results_rcv)
preds_rcv = pd.DataFrame(preds_rcv)
preds_rcv = preds_rcv.loc[:,0]
# preds_rcv
# 0       113667.776158


# In[ ]:


# BayesianRidge model
from sklearn import linear_model
# Algorithm - LassoCV
alpha_br = [1e-2, 1e-1, .2, .3, .4, .5, .6, .7, .9, 1]
model_br = make_pipeline(
                          MinMaxScaler(),
                          RobustScaler(),
                          BayesianRidge()
                        )
model_br.fit(train,y)
results_br = model_br.predict(test)
preds_br = np.expm1(results_br)
preds_br = pd.Series(preds_br)


# In[ ]:


# # Adding LinearRegression dropped us from .12135 to 0.12239
# # LinearRegression
# from sklearn.linear_model import LinearRegression  
# # Algorithm - LinearRegression
# model_lr = make_pipeline(
# #                            MinMaxScaler(),
#                            RobustScaler(),
#                            LinearRegression(
#                                    ))

# model_lr.fit(train,y)
# results_lr = model_lr.predict(test)
# preds_lr = np.expm1(results_lr)
# preds_lr = pd.DataFrame(preds_lr)
# preds_lr = preds_lr.loc[:,0]


# In[ ]:


# Testing XGB
# model_xgb = make_pipeline(
#     RobustScaler(),
#     XGBRegressor(n_estimators=1000, learning_rate=0.05))
# score_xgb = rmsle_cv(model_xgb)
# print('model_xgb')
# print(score_xgb.mean())
# model_xgb.fit(train,y)
# results_xgb = model_xgb.predict(test)
# results_xgb = np.expm1(results_xgb)
# predictions = results_xgb
predictions = (preds_encv + preds_lcv + preds_br + preds_rcv)/4
predictions


# In[ ]:


#  Creating competition submission file
# predictions = rmsle_cv(lasso)
# my_submission = pd.DataFrame({'Id': test_id, 'SalePrice': combinedpredictions})
my_submission = pd.DataFrame({'Id': test_id, 'SalePrice': predictions})
my_submission.to_csv('submission.csv',index=False)

