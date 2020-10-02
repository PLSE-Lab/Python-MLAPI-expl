#!/usr/bin/env python
# coding: utf-8

# * I could not get past the top 38%. If anyone has any correction or suggestion to make, feel welcome! 

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.columns


# ANALYZE VARIABLES DISTRIBUTION

# In[ ]:


train.SalePrice.describe()


# In[ ]:


sns.distplot(train.SalePrice)


# Most of the houses are between 100.000 and 300.000 dollars and is heavily right skewed.

# Analyze relationship with numeric values

# After an analysis of feature importance, some numeric features were chosen for a exploratory analysis:

# In[ ]:


numeric_columns = ['LotArea' , 'YearBuilt', 'GrLivArea', 'MiscVal', 'GarageArea']


# In[ ]:


y = train.SalePrice
for col in numeric_columns:
    x=train[col]
    sns.scatterplot(x,y)
    plt.title(col)
    plt.show()


# Analysing the graphs, LotArea doesn't seem to be as relevant as thought and there are a few outliers that will be treated.
# The same can be said about MiscVal.
# 
# As for GrLivArea and GarageArea, there is a strong positive correlation between it and Sale Price.
# 
# At last, we also perceive a slight correlation between Yearbuilt and SalePrice, especially on more recents years (>2000).

# In[ ]:


train.corr().SalePrice.sort_values(ascending=False)


# Analyze relationship with categoric values

# In[ ]:


categoric_columns = ['Neighborhood','BldgType','OverallQual','TotRmsAbvGrd']


y = train.SalePrice
for col in categoric_columns:
    x=train[col]
    sns.boxplot(x,y)
    plt.title(col)
    plt.show()


# Both OverallQual and TotRmsAbvGrd have a strong correlation with Sale Price, although the same cannot be said to BldgType and Neighborhood 

# In[ ]:


corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)


# With that matrix, we can notice some features are so correlated that might give the same information, for example: TotalBsmtSF/1stFlrSf, GarageCars/GarageArea, GarageYrBlt/YearBuilt and TotRmsAbvGrd/GrLivArea.

# Now, let's analyze the features with the strongest correlation with SalePrice

# In[ ]:


#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1)
fig, hm = plt.subplots(figsize=(10,8))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# Fill the missing Data

# In[ ]:


ntrain = train.shape[0]
ntest = test.shape[0]
y = train.SalePrice.values

full_df = pd.concat((train,test)).reset_index(drop=True)
full_df.drop(['SalePrice'], axis=1, inplace=True)


missing_values = full_df.isnull().sum().sort_values(ascending=False)

missing_pct = missing_values.loc[missing_values.values > 0]/len(full_df)
missing_pct


# The features missing more than 80% of the data will be deleted, as they do not add much value to the prediction.
# As for the other ones, the missing values will be filled with either the mode, for categorical features, or the mean. 

# In[ ]:


print(missing_pct.loc[missing_pct.values > 0.8].index)


# In[ ]:


missing_pct.index


# In[ ]:


drop_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence']

full_df.drop(drop_cols, axis=1, inplace=True)


# The reamining missing features will be either filled with 'None', '0' or the column's mode.

# In[ ]:


none_cols = ['FireplaceQu', 'GarageCond', 'GarageQual', 'GarageFinish', 'GarageType','BsmtCond', 'BsmtExposure', 
             'BsmtQual', 'BsmtFinType2', 'BsmtFinType1', 'MasVnrType', 'Exterior1st',
             'Exterior2nd']

for col in none_cols:
    full_df[col] = full_df[col].fillna('None')


# In[ ]:


missing_values = full_df.isnull().sum().sort_values(ascending=False)

missing_pct = missing_values.loc[missing_values.values > 0]/len(full_df)
missing_pct.index


# In[ ]:


zero_cols = ['MasVnrArea', 'BsmtHalfBath','BsmtFullBath','GarageCars','TotalBsmtSF', 
             'GarageArea', 'BsmtUnfSF','BsmtFinSF2', 'BsmtFinSF1','LotFrontage']

for col in zero_cols:
    full_df[col] = full_df[col].fillna(0)
    


# In[ ]:


mode_cols = ['MSZoning', 'Functional', 'Utilities', 'SaleType',
             'KitchenQual', 'Electrical']

for col in mode_cols:
    mode = full_df[col].mode()
    full_df[col] = full_df[col].fillna(mode[0])


# In[ ]:


id_na = list(full_df.loc[full_df['GarageYrBlt'].isna()].Id.values)

for row in id_na:
    full_df.loc[row-1,'GarageYrBlt'] = full_df.loc[row-1,'YearBuilt']


# In[ ]:


missing_values = full_df.isnull().sum().sort_values(ascending=False)

missing_pct = missing_values.loc[missing_values.values > 0]/len(full_df)
missing_pct


# FEATURE ENGINEERING

# In[ ]:


full_df['TotalBath'] = full_df['BsmtFullBath'] + full_df['BsmtHalfBath'] + full_df['FullBath'] + full_df['HalfBath']
full_df['TotalArea'] = full_df['TotalBsmtSF'] + full_df['1stFlrSF'] + full_df['2ndFlrSF']
full_df['YrBltAndRemod']=full_df['YearBuilt']+full_df['YearRemodAdd']


# In[ ]:


full_df['MSSubClass'] = full_df['MSSubClass'].apply(str)

full_df['MoSold'] = full_df['MoSold'].astype(str)


# In[ ]:


full_df.columns


# In[ ]:


drop_cols = ['LotFrontage','BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 
             'PoolArea', 'GarageCars','LandSlope', 'MoSold', 'TotalBsmtSF', 'MiscVal',
             'HouseStyle', 'RoofMatl','Condition2'
            ]

full_df.drop(drop_cols, axis=1, inplace=True)


# In[ ]:


full_df = pd.get_dummies(full_df,drop_first=True)


# In[ ]:


df_train = full_df[:ntrain]
df_test = full_df[ntrain:]

test_id = test['Id']
df_train.set_index('Id',inplace=True)
df_test.set_index('Id',inplace=True)


# In[ ]:


df_train = pd.get_dummies(df_train,drop_first=True)
df_test = pd.get_dummies(df_test,drop_first=True)


# In[ ]:


print(df_train.shape)
print(df_test.shape)


# MODEL

# In[ ]:


from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_log_error
from math import log
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# In[ ]:


random_state=42

x_train, x_val, y_train,y_val = train_test_split(df_train, y, random_state=42)


# In[ ]:


estimator = XGBRegressor(objective='reg:squarederror')

params = {
        'max_depth':range(3,7,2),
        'min_child_weight':range(1,5,2)
         }

    
def tuning(estimator, params):
    grid = GridSearchCV(estimator, param_grid = params, scoring='neg_mean_squared_log_error')
    grid.fit(x_train,y_train)
    print(grid.best_params_)
    print(-grid.best_score_)
tuning(estimator,params)


# In[ ]:


estimator = XGBRegressor(objective='reg:squarederror',max_depth = 3, min_child_weight = 1)

params = {
        'gamma':[i/10.0 for i in range(0,5)]
         }

tuning(estimator, params)


# In[ ]:


estimator = XGBRegressor(objective='reg:squarederror',max_depth = 3, min_child_weight = 1,
                         gamma = 0)

params = {
        'learning_rate' : [0.01,0.03,0.1,0.3]
         }

tuning(estimator, params)


# In[ ]:


estimator = XGBRegressor(objective='reg:squarederror',max_depth =3, min_child_weight = 1)

estimator.fit(x_train,y_train, 
             eval_set=[(x_val, y_val)], verbose=False)

y_pred = estimator.predict(x_val)

print(mean_squared_log_error(y_pred,y_val))

feat_imp = pd.Series(estimator.feature_importances_)


# In[ ]:


feat = pd.concat([feat_imp,pd.DataFrame(df_train.columns)],axis=1)
feat.columns = ['Importance','Columns']
feat = feat.sort_values(by = 'Importance',ascending=False)


# In[ ]:


feat.head(25)


# In[ ]:


feat.tail(60)

