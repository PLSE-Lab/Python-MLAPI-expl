# -*- coding: utf-8 -*-
"""
Created on Sat May 13 14:57:02 2017

@author: Angela
"""

import csv
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from random import randint, randrange, uniform
from scipy import stats
from sklearn import preprocessing, ensemble
import scipy.stats as st
from sklearn.grid_search import RandomizedSearchCV
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from xgboost.sklearn import XGBRegressor

NOMINAL_COLS = ['BsmtExposure','LandContour','MoSold','MSSubClass','MSZoning','Alley','LotConfig','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation','Heating','Electrical','GarageType','Fence','MiscFeature','SaleCondition','SaleType']
QUAL_COLS = ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond','PoolQC']
QUAL_ORDR = ['NA','Po','Fa','TA','Gd','Ex']
ORDINALS = {'LotShape':["Reg","IR1","IR2","IR3"],'LandSlope':["Gtl","Mod","Sev"],
                'BsmtFinType1':["NA","Unf","LwQ","Rec","BLQ","ALQ","GLQ"],'BsmtFinType2':["NA","Unf","LwQ","Rec","BLQ","ALQ","GLQ"],
                'Functional':["Sal","Sev","Maj2","Maj1","Mod","Min2","Min1","Typ"],'PavedDrive':["N","P","Y"],
                'GarageFinish':["NA","Unf","RFn","Fin"], 'CentralAir':['N','Y'],'Street':['Gravel','Pave']}
QUANT_COLS = ['LotFrontage', 'LotArea', 'LotShape','LandSlope', 'OverallQual', 'OverallCond', 
           'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'ExterQual', 
            'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtFinType1',
            'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'HeatingQC',
            '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
            'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Fireplaces',
            'FireplaceQu', 'GarageYrBlt', 'GarageFinish',  'GarageCars', 'GarageArea', 'GarageQual',
            'GarageCond', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
            'PoolQC', 'MiscVal', 'YrSold', 'Median_Household_Income', 'agegroupA','agegroupB','agegroupC','agegroupD']
TO_REMOVE = ['Id','Utilities']

def prep(df_train, df_test):
    # Change cat and ordinal to quantitative
    df_train = mod_nbhd(df_train)
    print(df_train.columns.values)
    df_test = mod_nbhd(df_test)
    
    df_train = prep_h(df_train)
    df_test = prep_h(df_test)
    
    # Fill columns with means
    df_train = df_train.fillna(df_train.mean())
    df_test = df_test.fillna(df_train.mean())
    
   # Remove columns that are only in train or only in test
    remove_missing_cols(df_train, df_test)
    
    # Normalize skewed
    normalize(df_train, df_test)
    # Scale
    scale(df_train, df_test)
   
    return df_train, df_test

def prep_h(df):
    df.drop(TO_REMOVE, axis=1,inplace=True)
    df = pd.get_dummies(df,columns=NOMINAL_COLS)
    
    for col in QUAL_COLS:
        df[col] = pd.Categorical(df[col], categories=QUAL_ORDR, ordered=True)
        df[col] = df[col].cat.codes + 1

    for key in ORDINALS:
        df[key] = pd.Categorical(df[key], categories=ORDINALS[key], ordered=True)
        df[key] = df[key].cat.codes + 1

    return df

def scale(df_train, df_test):
    scaler = preprocessing.StandardScaler()
    df_train[QUANT_COLS] = scaler.fit_transform(df_train[QUANT_COLS])
    df_test[QUANT_COLS] = scaler.transform(df_test[QUANT_COLS])

def intersect(a, b):
    return list((set(a)).intersection(set(b)))
    
def normalize(df_train, df_test):
    # https://www.kaggle.com/pmarcelino/house-prices-advanced-regression-techniques/comprehensive-data-exploration-with-python
    # https://www.kaggle.com/juliencs/house-prices-advanced-regression-techniques/a-study-on-regression-applied-to-the-ames-dataset
    skewness = df_train[QUANT_COLS].apply(lambda x: st.skew(x))
    skewness = skewness[abs(skewness) > 0.5]
    skewed_features = intersect(skewness.index, df_train.columns.values) 
    print (skewed_features)
    df_train[skewed_features] = np.log1p(df_train[skewed_features])
    df_test[skewed_features] = np.log1p(df_test[skewed_features])

def remove_missing_cols(train, test):
    cols_in_train = set(train.columns.values) - set(test.columns.values)
    cols_in_test = set(test.columns.values) - set(train.columns.values)
    train.drop([x for x in cols_in_train if x != 'SalePrice'], axis = 1, inplace = True)
    test.drop(cols_in_test, axis = 1, inplace = True)
    return train, test
    
# Adds new columns to df based on Neighborhood.
def mod_nbhd(df):
    # Column names here. For example, 
    #   headers = ['Median_Income', 'Income_per_Capita', 'Kid_Population', 'Adult_Population']
    headers = ['Median_Household_Income','agegroupA','agegroupB','agegroupC','agegroupD']
    
    # Replacement information here in the form,
    #   data = { neighborhood : [new_col_1_value, new_col_2_value, ..., new_col_n_value], ...}
    # For example,
    #   data = { 'NWAmes'   : [ 100000 , 50000 , .3, .7],
    #            'Veenker'  : [ 58000, 50000, .1, .9] }
    data = {'Blmngtn'   : [103608, float(778/2926), float(391/2926), float(1040/2926), float(717/2926)],
            'Blueste'   : [36940, float(12/691), float(381/691), float(143/691), float(155/691)],
            'BrDale'	: [103608, float(778/2926), float(391/2926), float(1040/2926), float(717/2926)],
            'BrkSide'   : [61793, float(179/1326), float(421/1326), float(543/1326), float(183/1326)],
            'ClearCr'   : [49038, float(539/1807), float(205/1807), float(808/1807), float(255/1807)],
            'CollgCr'	: [55769, float(293/1748), float(857/1748), float(598/1748), float(0/1748)],
            'Crawfor'	: [58952, float(173/1761), float(576/1761), float(667/1761), float(345/1761)],
            'Edwards'	: [39424, float(228/2987), float(2100/2987), float(432/2987), float(227/2987)],
            'Gilbert'	: [103608, float(778/2926), float(391/2926), float(1040/2926), float(717/2926)],
            'IDOTRR'	: [15132, float(194/14001) , float(13189/14001), float(544/14001), float(74/14001)],
            'MeadowV'	: [50313, float(678/3177), float(1221/3177), float(1002/3177), float(276/3177)],
            'Mitchel'	: [41250, float(31/128), float(21/128), float(46/128), float(30/128)],
            'NAmes'	    : [52395, float(1236/5800), float(1577/5800), float(1862/5800), float(1125/5800)],
            'NoRidge'	: [83587, float(827/3194), float(581/3194), float(1467/3194), float(319/3194)],
            'NPkVill'	: [41875, float(142/943), float(365/943), float(246/943), float(190/943)],
            'NridgHt'	: [103608, float(778/2926), float(391/2926), float(1040/2926), float(717/2926)],
            'NWAmes'	: [92361, float(285/1401), float(366/1401), float(469/1401), float(281/1401)],
            'OldTown'	: [34583, float(49/551), float(206/551), float(239/551), float(57/551)],
            'SWISU'	    : [15132, float(194/14001), float(13189/14001), float(544/14001), float(74/14001)],
            'Sawyer'	: [74393, float(498/1975), float(561/1975), float(713/1975), float(203/1975)],
            'SawyerW'	: [54975, float(575/2858), float(1244/2858), float(857/2858), float(182/2858)],
            'Somerst'	: [83587, float(827/3194), float(581/3194), float(1467/3194), float(319/3194)],
            'StoneBr'	: [103608, float(778/2926), float(391/2926), float(1040/2926), float(717/2926)],
            'Timber'	: [58952, float(173/1761), float(576/1761), float(667/1761), float(345/1761)],
            'Veenker'	: [83587, float(827/3194), float(581/3194), float(1467/3194), float(319/3194)]
            }
          
    df = moddf(df, 'Neighborhood', headers, data)
    return df

# Adds columns to df based on row value of df[col_name]
def add_columns(row, col_name, replacement, index):
    return replacement[row[col_name]][index]

# Modifies df: 1. adds headers to df, 2. adds columns based on df[col] value per row. The new 
# column values are based on the df[col] values and the replacement as found in replacement.
def moddf(df, col, headers, replacement):
    for h in range(len(headers)):
        df[headers[h]] = df.apply(lambda row: add_columns(row, col, replacement, h),axis=1)
    return df

def format_output(ids,prediction):
    with open('./house_prices_output.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Id','SalePrice'])
        for i in range(len(prediction)):
            writer.writerow([ids[i], float(prediction[i])])
    
df_train_raw = pd.read_csv("../input/train.csv")

df_test_raw = pd.read_csv("../input/test.csv")
IDs = df_test_raw['Id'].values
df_train, df_test = prep(df_train_raw, df_test_raw)
train_x_og = df_train.ix[:, df_train.columns != 'SalePrice'].as_matrix()
train_y_og = np.log1p(df_train['SalePrice'])

test_x = df_test.as_matrix()

isol = ensemble.IsolationForest(contamination=0.05)
isol.fit(train_x_og,train_y_og)
outliers = isol.predict(train_x_og)
indices = []
for i in range(len(outliers)):
    if outliers[i] == -1:
        indices.append(i)

df_train.drop(indices, inplace=True)

train_x = df_train.ix[:, df_train.columns != 'SalePrice'].as_matrix()
train_y = np.log1p(df_train['SalePrice'].as_matrix())

clf = XGBRegressor(n_estimators=600,learning_rate=.05,subsample=0.8,colsample_bytree=0.8,max_depth=3,min_child_weight=3)
#clf = XGBRegressor(n_estimators=200,learning_rate=.05,max_depth=8,min_child_weight=6)
#clf.fit(train_x, train_y)

scores = cross_val_score(clf,train_x,train_y,scoring="neg_mean_squared_error",cv=5)
print (np.sqrt(np.abs(scores)))
#print ('Best params:',clf.best_params_)

print (train_x.shape,train_x_og.shape)

reg2 = LassoCV()
reg2.fit(train_x,train_y)
print (reg2.score(train_x_og,train_y_og))

ridge = RidgeCV()
ridge.fit(train_x, train_y)
print (ridge.score(train_x_og, train_y_og))

clf.fit(train_x,train_y)
# print (clf.score(train_x_og, train_y_og))
test_predict = clf.predict(test_x)
result = np.expm1(test_predict)
format_output(IDs,result)
print (result)
