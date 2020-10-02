#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


pd.set_option('display.max_columns',None)


# In[ ]:


train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


combine_data = pd.concat([train_data, test_data])


# In[ ]:


combine_data.shape


# In[ ]:


combine_data.tail()


# # 1. Missing value Imputation

# In[ ]:


top_null_columns = combine_data.isnull().sum() / combine_data.shape[0] * 100
# top_null_columns.values > 20
deleted_null_columns = top_null_columns[top_null_columns.values > 20].keys()
deleted_null_columns


# In[ ]:


af_del_combine = combine_data.drop(deleted_null_columns, axis='columns')
af_del_combine.head()


# In[ ]:


plt.figure(figsize=(20, 7))
sns.heatmap(af_del_combine.isnull())


# ## 1.1 Numerical Imputation

# In[ ]:


numeric_data = af_del_combine.select_dtypes(['int64', 'float64'])
numeric_data_columns = numeric_data.columns
numeric_data_columns


# In[ ]:


numeric_null =  numeric_data.isnull().sum()
numeric_null_collumns = numeric_null[numeric_null.values > 0].keys()
numeric_null_collumns


# In[ ]:


numeric_null


# In[ ]:


numeric_data[numeric_data[numeric_null_collumns].isnull().any(axis=1)].head()


# In[ ]:


numeric_fill_data_mean = numeric_data.fillna(numeric_data.mean())


# In[ ]:


numeric_fill_data_mean.isnull().sum().sum()


# ## 1.2 Categorical Imputation

# In[ ]:


categorical_data = af_del_combine.select_dtypes(['O'])
categorical_data.head()


# In[ ]:


cat_null_col = categorical_data.isnull().sum()
cat_null_col = cat_null_col[cat_null_col.values > 0].keys()
cat_null_col


# In[ ]:


categorical_data[categorical_data[cat_null_col].isnull().any(axis=1)].head()


# In[ ]:


categorical_fill_data_mode = categorical_data.copy()


# In[ ]:





# In[ ]:


categorical_fill_data_mode.head()


# In[ ]:


# categorical_data['GarageType'].mode()[0]


# In[ ]:


# categorical_data['GarageType'].fillna(categorical_data['GarageType'].mode)


# In[ ]:


for column in cat_null_col:
    print(column, categorical_data[column].mode()[0])
    categorical_fill_data_mode[column] = categorical_data[column].fillna(categorical_data[column].mode()[0])


# In[ ]:


categorical_data.isnull().sum().sum()


# In[ ]:


categorical_fill_data_mode.isnull().sum().sum()


# In[ ]:


combine_data.head()


# In[ ]:


combine_data.isnull().sum()


# In[ ]:


numeric_fill_data_mean.head()


# In[ ]:


categorical_fill_data_mode.head()


# In[ ]:


combine_fill_data = pd.concat([numeric_fill_data_mean, categorical_fill_data_mode], axis=1, sort=False)


# In[ ]:


combine_fill_data.head()


# In[ ]:


combine_fill_data.isnull().sum().sum()


# # 2. One-Hot Encoding

# In[ ]:


categorical_columns = categorical_data.columns
categorical_columns


# In[ ]:


combine_fill_data[categorical_columns].head()


# In[ ]:


combine_fill_data['GarageCond'].unique()


# In[ ]:


ordinal_cat_columns = ['KitchenQual', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'GarageQual', 'GarageCond']
nominal_cat_columns = ['BldgType',
 'Condition1',
 'Condition2',
 'Exterior1st',
 'Electrical',
 'GarageType',
 'GarageFinish',
 'SaleCondition',
 'RoofMatl',
 'SaleType',
 'MasVnrType',
 'LandSlope',
 'LotShape',
 'PavedDrive',
 'Utilities',
 'Heating',
 'Functional',
 'LandContour',
 'LotConfig',
 'Exterior2nd',
 'Neighborhood',
 'HouseStyle',
 'Street',
 'MSZoning',
 'Foundation',
 'RoofStyle',
 'CentralAir'
]


# In[ ]:


len(nominal_cat_columns)


# In[ ]:


len(ordinal_cat_columns)


# In[ ]:


# pd.get_dummies()


# In[ ]:


# combine_data['BsmtCond'].unique()


# In[ ]:


combine_fill_data['GarageCond'].unique()


# In[ ]:


# combine_fill_data['KitchenQual']


# In[ ]:


combine_map_data = combine_fill_data.copy()


# In[ ]:


KitchenQual_map = {'Gd':3, 'TA':2, 'Ex':4, 'Fa':1}
ExterQual_map = {'Gd':3, 'TA':2, 'Ex':4, 'Fa':1}
ExterCond_map = {'TA': 3, 'Gd': 4, 'Fa': 2, 'Po': 1, 'Ex': 5}
BsmtQual_map = {'Gd':3, 'TA':2, 'Ex':4, 'Fa':1}
BsmtCond_map = {'TA':3, 'Gd':4, 'Fa':2, 'Po':1}
BsmtExposure_map = {'No': 1, 'Gd': 4, 'Mn': 2, 'Av': 3}
BsmtFinType1_map = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf':1}
BsmtFinType2_map = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf':1} 
HeatingQC_map = {'TA': 3, 'Gd': 4, 'Fa': 2, 'Po': 1, 'Ex': 5}
GarageQual_map = {'TA': 3, 'Gd': 4, 'Fa': 2, 'Po': 1, 'Ex': 5}
GarageCond_map = {'TA': 3, 'Gd': 4, 'Fa': 2, 'Po': 1, 'Ex': 5}


# In[ ]:


combine_map_data['KitchenQual'] = combine_fill_data['KitchenQual'].map(KitchenQual_map)
combine_map_data['ExterQual'] = combine_fill_data['ExterQual'].map(ExterQual_map)
combine_map_data['ExterCond'] = combine_fill_data['ExterCond'].map(ExterCond_map)
combine_map_data['BsmtQual'] = combine_fill_data['BsmtQual'].map(BsmtQual_map)
combine_map_data['BsmtCond'] = combine_fill_data['BsmtCond'].map(BsmtCond_map)
combine_map_data['BsmtExposure'] = combine_fill_data['BsmtExposure'].map(BsmtExposure_map)
combine_map_data['BsmtFinType1'] = combine_fill_data['BsmtFinType1'].map(BsmtFinType1_map)
combine_map_data['BsmtFinType2'] = combine_fill_data['BsmtFinType2'].map(BsmtFinType2_map)
combine_map_data['HeatingQC'] = combine_fill_data['HeatingQC'].map(HeatingQC_map)
combine_map_data['GarageQual'] = combine_fill_data['GarageQual'].map(GarageQual_map)
combine_map_data['GarageCond'] = combine_fill_data['GarageCond'].map(GarageCond_map)


# In[ ]:


combine_map_data['GarageCond']


# In[ ]:


combine_map_data.head()


# In[ ]:


combine_dummy_data = pd.get_dummies(combine_map_data, drop_first=True)
combine_dummy_data.head()


# In[ ]:


combine_dummy_data[combine_dummy_data['CentralAir_Y'] == 0].head()


# # scaling dataset

# In[ ]:


from sklearn.preprocessing import RobustScaler


# In[ ]:


saleprice = np.log(train_data['SalePrice']+1)
saleprice


# In[ ]:


combine_dummy_drop_data = combine_dummy_data.drop(['Id'], axis=1)
combine_dummy_drop_data


# In[ ]:


robust = RobustScaler()
robust.fit(combine_dummy_drop_data)
combine_scale_data = robust.transform(combine_dummy_drop_data)


# In[ ]:


combine_scale_data[:100]


# In[ ]:


combine_dummy_drop_data.columns


# In[ ]:


combine_column_scale_data = pd.DataFrame(combine_scale_data, columns=combine_dummy_drop_data.columns)


# In[ ]:


combine_column_scale_data


# # Machine Learning Models

# In[ ]:


train_len = len(train_data)
train_len


# In[ ]:


X_train = combine_scale_data[:train_len]
X_test = combine_scale_data[train_len:]
y_train = saleprice


# In[ ]:


from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, r2_score


# In[ ]:


from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, r2_score

def test_model(model, X_train=X_train, y_train=y_train):
    cv = KFold(n_splits = 3, shuffle=True, random_state = 45)
    r2 = make_scorer(r2_score)
    r2_val_score = cross_val_score(model, X_train, y_train, cv=cv, scoring = r2)
    score = [r2_val_score.mean()]
    return score


# # Linear Regression

# In[ ]:


import sklearn.linear_model as linear_model
LR = linear_model.LinearRegression()
test_model(LR)


# # Support Vector Machine

# In[ ]:


from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf')
test_model(svr_reg)


# # Decision Tree Regressor

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state=21)
test_model(dt_reg)


# # Random Forest Regressor

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 1000, random_state=51)
test_model(rf_reg)


# # Bagging & boosting

# In[ ]:


from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
br_reg = BaggingRegressor(n_estimators=1000, random_state=51)
gbr_reg = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, loss='ls', random_state=51)


# In[ ]:


test_model(br_reg)


# In[ ]:


test_model(gbr_reg)


# # XGBoost

# In[ ]:


import xgboost
#xgb_reg=xgboost.XGBRegressor()
xgb_reg = xgboost.XGBRegressor(bbooster='gbtree', random_state=51)
test_model(xgb_reg)


# In[ ]:





# In[ ]:


gbr_reg.fit(X_train, y_train)


# In[ ]:


y_pred = np.exp(gbr_reg.predict(X_test)).round(2)


# In[ ]:


y_pred


# In[ ]:


submit_result = pd.concat([test_data['Id'],pd.DataFrame(y_pred)], axis=1)
submit_result.columns=['Id', 'SalePrice']


# In[ ]:


submit_result.head(10)


# In[ ]:


submit_result.to_csv('house_submission.csv', index=False )


# note: this notebook still in progress. work on increasing accuracy, added soon acknowledgement and description. thank you.
