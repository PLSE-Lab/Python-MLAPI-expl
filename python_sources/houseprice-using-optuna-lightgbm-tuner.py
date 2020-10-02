#!/usr/bin/env python
# coding: utf-8

# # What is Optuna?
# LightGBM is one of powerful tool. And take a lot of time to tuning (for noob like me)
# Optuna LightGBM Tuner (https://github.com/optuna/optuna/pull/549) can find out the best parameter on LightGBM.
# Here I show you one of a cheap code using this powerful tool. thanks.

# In[ ]:


# import lightbgm as lab (you just import as follows instead of this code)
from optuna.integration import lightgbm as lgb


# In[ ]:


# from tutorial
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.head()


# In[ ]:


test.head()


# # Categorical variables
# Label Encoding some categorical variables that may contain information in their ordering set

# In[ ]:


numeric_features = train.select_dtypes(include=[np.number])
numeric_features.columns


# In[ ]:


categorical_features = train.select_dtypes(include=[np.object])
categorical_features.columns


# to check some categorical variables

# In[ ]:


train_test['SaleCondition'].unique()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
cols = ('MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
       'SaleType', 'SaleCondition')

cols_defaults = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(train[c].values)) 
    train[c] = lbl.transform(list(train[c].values))

# shape        
print('Shape train: {}'.format(train.shape))


# In[ ]:


# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(test[c].values)) 
    test[c] = lbl.transform(list(test[c].values))

# shape        
print('Shape test: {}'.format(test.shape))


# to confirm categorical variables has converted.

# In[ ]:


train_test['SaleCondition'].unique()


# In[ ]:


train_test=pd.concat([train,test],axis=0,sort=False)
train_test.head()


# In[ ]:


train.isnull().sum() 


# In[ ]:


train_test.groupby('SaleCondition').count()['Id']


# In[ ]:


train['SalePrice'].hist(bins = 40)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sb
correlation_train=train.corr()
sb.set(font_scale=2)
plt.figure(figsize=(50,50))
ax = sb.heatmap(correlation_train, annot=True,annot_kws={"size": 25},fmt='.1f', linewidths=.5)


# In[ ]:


correlation_train.columns


# # Feature Engineering
# 
# pick up low-correlation elements, under 0.1 or over -0.1.

# In[ ]:


corr_dict=correlation_train['SalePrice'].sort_values(ascending=False).to_dict()
unimportant_columns=[]
for key,value in corr_dict.items():
    if  ((value>=-0.1) & (value<=0)) | (value==0.1):
        unimportant_columns.append(key)
unimportant_columns


# # Drop'em up
# 

# In[ ]:


train2= train.drop(['BsmtFinSF2',
 'Utilities',
 'BsmtHalfBath',
 'MiscVal',
 'Id',
 'LowQualFinSF',
 'YrSold',
 'SaleType',
 'LotConfig',
 'OverallCond',
 'MSSubClass',
 'BldgType',
 'Heating'],axis=1)


# In[ ]:


correlation_train2=train2.corr()
sb.set(font_scale=2)
plt.figure(figsize=(50,50))
ax = sb.heatmap(correlation_train2, annot=True,annot_kws={"size": 25},fmt='.1f',cmap='PiYG', linewidths=.5)


# # Add non-AI knowledge
# 
# See heatmap, find high-correlations between features which seems to be same feature (i,e, 'GarageCars','GareageArea' they are have 0.9)
# I dropped three features (tagged # del).

# In[ ]:


train2.columns


# In[ ]:


train3= train.loc[:,['MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape',
       'LandContour', 'LandSlope', 
        #'Neighborhood',
        'Condition1', 'Condition2',
       'HouseStyle', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',
       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
       'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
       'BsmtUnfSF', 'TotalBsmtSF', 'HeatingQC', 'CentralAir', 'Electrical',
       #'1stFlrSF', 
        '2ndFlrSF', 'GrLivArea', 'BsmtFullBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       #'TotRmsAbvGrd',
       'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       #'GarageYrBlt',
       'GarageFinish', 
       #'GarageCars', 
       'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 
       #'PoolArea', 
        'PoolQC',
       'Fence', 'MiscFeature', 
       #'MoSold', 
       'SaleCondition', 'SalePrice']]


# In[ ]:


correlation_train3=train3.corr()
sb.set(font_scale=2)
plt.figure(figsize=(50,50))
ax = sb.heatmap(correlation_train3, annot=True,annot_kws={"size": 25},fmt='.1f',cmap='PiYG', linewidths=.5)


# In[ ]:


corr_dict3=correlation_train3['SalePrice'].sort_values(ascending=False).to_dict()
unimportant_columns_train3=[]
for key,value in corr_dict3.items():
    if  ((value>=-0.1) & (value<=0)) | (value==0.1):
        unimportant_columns_train3.append(key)
unimportant_columns_train3


# train3 has only important columns (val < 0.1 or val > 0.1)

# In[ ]:


import seaborn as sns
from sklearn.linear_model import LinearRegression

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:



sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea','EnclosedPorch','BsmtUnfSF']
sns.pairplot(train3[cols], size = 2.5)
plt.show()


# #  It's time to go

# In[ ]:


test= test.loc[:,['MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape',
       'LandContour', 'LandSlope', 
        #'Neighborhood',
        'Condition1', 'Condition2',
       'HouseStyle', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',
       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
       'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
       'BsmtUnfSF', 'TotalBsmtSF', 'HeatingQC', 'CentralAir', 'Electrical',
       #'1stFlrSF', 
        '2ndFlrSF', 'GrLivArea', 'BsmtFullBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       #'TotRmsAbvGrd',
       'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       #'GarageYrBlt',
       'GarageFinish', 
       #'GarageCars', 
       'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 
       #'PoolArea', 
        'PoolQC',
       'Fence', 'MiscFeature', 
       #'MoSold', 
       'SaleCondition'
                 ]]


# In[ ]:


len(test)


# In[ ]:


len(train3)


# In[ ]:


X_train = train3.drop([ 'SalePrice'], axis=1)
y_train = np.log1p(train3['SalePrice'])


# In[ ]:


from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

y_preds = []
models = []
oof_train = np.zeros((len(X_train),))
cv = KFold(n_splits=5, shuffle=True, random_state=0)


params = {
    'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
}

for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train)):
    X_tr = X_train.loc[train_index, :]
    X_val = X_train.loc[valid_index, :]
    y_tr = y_train[train_index]
    y_val = y_train[valid_index]

    lgb_train = lgb.Dataset(X_tr,
                            y_tr,
                            #categorical_feature=categorical_cols
                           )

    lgb_eval = lgb.Dataset(X_val,
                           y_val,
                           reference=lgb_train,
                           #categorical_feature=categorical_cols
                          )

    model = lgb.train(params,
                      lgb_train,
                      valid_sets=[lgb_train, lgb_eval],
                      verbose_eval=10,
                      num_boost_round=1000,
                      early_stopping_rounds=10)


    oof_train[valid_index] = model.predict(X_val,
                                           num_iteration=model.best_iteration)
    y_pred = model.predict(test,
                           num_iteration=model.best_iteration)

    y_preds.append(y_pred)
    models.append(model)


# In[ ]:


print(f'CV: {np.sqrt(mean_squared_error(y_train, oof_train ))}')


# In[ ]:


sub = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
y_sub = sum(y_preds) / len(y_preds)
y_sub = np.expm1(y_sub)
sub['SalePrice'] = y_sub
sub.to_csv('submission.csv', index=False)
sub.head()


# I tried basical parameter, put train2( drop three features) previously.
# LightGBM Tuner was the highest score compare to them,
# # 0.13056:LightGBM on train3 with categorical valiables
# # 0.14281:LightGBM on train3 only numerical valiables
# # 0.14357:LightGBM with basical paremeter
# # 0.14498:LightGBM on train2 (autoparam and high-corelation features)
# 
# Thanks
