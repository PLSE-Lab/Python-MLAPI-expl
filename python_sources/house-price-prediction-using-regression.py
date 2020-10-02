#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[ ]:


import numpy as np 
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import plotly_express as px

# color_pallete = ['#FF1744', '#666666']
# sns.set_palette(color_pallete, 2)
plt.style.use('fivethirtyeight')

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import eli5
from eli5.sklearn import PermutationImportance


# # Column names

# In[ ]:


# SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
# MSSubClass: The building class
# MSZoning: The general zoning classification
# LotFrontage: Linear feet of street connected to property
# LotArea: Lot size in square feet
# Street: Type of road access
# Alley: Type of alley access
# LotShape: General shape of property
# LandContour: Flatness of the property
# Utilities: Type of utilities available
# LotConfig: Lot configuration
# LandSlope: Slope of property
# Neighborhood: Physical locations within Ames city limits
# Condition1: Proximity to main road or railroad
# Condition2: Proximity to main road or railroad (if a second is present)
# BldgType: Type of dwelling
# HouseStyle: Style of dwelling
# OverallQual: Overall material and finish quality
# OverallCond: Overall condition rating
# YearBuilt: Original construction date
# YearRemodAdd: Remodel date
# RoofStyle: Type of roof
# RoofMatl: Roof material
# Exterior1st: Exterior covering on house
# Exterior2nd: Exterior covering on house (if more than one material)
# MasVnrType: Masonry veneer type
# MasVnrArea: Masonry veneer area in square feet
# ExterQual: Exterior material quality
# ExterCond: Present condition of the material on the exterior
# Foundation: Type of foundation
# BsmtQual: Height of the basement
# BsmtCond: General condition of the basement
# BsmtExposure: Walkout or garden level basement walls
# BsmtFinType1: Quality of basement finished area
# BsmtFinSF1: Type 1 finished square feet
# BsmtFinType2: Quality of second finished area (if present)
# BsmtFinSF2: Type 2 finished square feet
# BsmtUnfSF: Unfinished square feet of basement area
# TotalBsmtSF: Total square feet of basement area
# Heating: Type of heating
# HeatingQC: Heating quality and condition
# CentralAir: Central air conditioning
# Electrical: Electrical system
# 1stFlrSF: First Floor square feet
# 2ndFlrSF: Second floor square feet
# LowQualFinSF: Low quality finished square feet (all floors)
# GrLivArea: Above grade (ground) living area square feet
# BsmtFullBath: Basement full bathrooms
# BsmtHalfBath: Basement half bathrooms
# FullBath: Full bathrooms above grade
# HalfBath: Half baths above grade
# Bedroom: Number of bedrooms above basement level
# Kitchen: Number of kitchens
# KitchenQual: Kitchen quality
# TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
# Functional: Home functionality rating
# Fireplaces: Number of fireplaces
# FireplaceQu: Fireplace quality
# GarageType: Garage location
# GarageYrBlt: Year garage was built
# GarageFinish: Interior finish of the garage
# GarageCars: Size of garage in car capacity
# GarageArea: Size of garage in square feet
# GarageQual: Garage quality
# GarageCond: Garage condition
# PavedDrive: Paved driveway
# WoodDeckSF: Wood deck area in square feet
# OpenPorchSF: Open porch area in square feet
# EnclosedPorch: Enclosed porch area in square feet
# 3SsnPorch: Three season porch area in square feet
# ScreenPorch: Screen porch area in square feet
# PoolArea: Pool area in square feet
# PoolQC: Pool quality
# Fence: Fence quality
# MiscFeature: Miscellaneous feature not covered in other categories
# MiscVal: $Value of miscellaneous feature
# MoSold: Month Sold
# YrSold: Year Sold
# SaleType: Type of sale
# SaleCondition: Condition of sale


# # Datasets

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


print("Train shape : ", train.shape, "\nTest shape : ", test.shape)


# In[ ]:


train.head()


# In[ ]:


# train.columns


# In[ ]:


# train.info()


# In[ ]:


# train.describe(include='all')


# In[ ]:


# train.isnull().sum()


# In[ ]:


# # correlation heatmap

# plt.figure(figsize=(40, 40))
# sns.heatmap(train.corr(), annot=True, cmap='RdBu', vmax=1, vmin=-1)
# plt.plot()


# In[ ]:


train['SalePrice'].describe()


# In[ ]:


# sns.distplot(train['SalePrice'])


# In[ ]:


# finding numerical and categorical columns

cols = train.drop(['Id', 'SalePrice'], axis=1).columns

num_cols = []
cat_cols = []

for i in train[cols]:
    if(train[i].dtype=='object'):
        cat_cols.append(i)
    else:
        num_cols.append(i)
        
# print(num_cols)
# print()
# print(cat_cols)


# In[ ]:


for i in num_cols:
    data = pd.concat([train['SalePrice'], train[i]], axis=1)
    data.plot.scatter(x=i, y='SalePrice', ylim=(0,800000));


# In[ ]:


for i in cat_cols:
    data = pd.concat([train['SalePrice'], train[i]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=i, y="SalePrice", data=train)
    fig.axis(ymin=0, ymax=800000);


# In[ ]:


null_cols = []

for i in train.isnull().sum().index:
    if train.isnull().sum()[i] > 0:
        null_cols.append(i)
        
for i in test.isnull().sum().index:
    if (test.isnull().sum()[i] > 0) and (i not in null_cols):
        null_cols.append(i)
        
print(null_cols)


# In[ ]:





# In[ ]:


train[null_cols].describe(include='all')


# In[ ]:


plt.figure(figsize=(8, 8))
sns.heatmap(train[[i for i in null_cols if i in num_cols]].corr(), annot=True, cmap='RdBu', vmax=1, vmin=-1)
plt.plot()


# In[ ]:


for df in [train, test]:
    for i in num_cols:
        avg = df[i].mean()
        df[i] = df[i].fillna(avg)


# In[ ]:


null_cols = []

for i in train.isnull().sum().index:
    if train.isnull().sum()[i] > 200:
        null_cols.append(i)
        
for i in test.isnull().sum().index:
    if (test.isnull().sum()[i] > 200) and (i not in null_cols):
        null_cols.append(i)
        
print(null_cols)
print()

for i in [train, test]:
    print(i.shape)
    i.drop(null_cols, axis=1, inplace=True)
    print(i.shape)
    print()


# In[ ]:


train = train.fillna(df.mode().iloc[0])
test = test.fillna(df.mode().iloc[0])


# In[ ]:


cols = train.drop(['Id', 'SalePrice'], axis=1).columns

num_cols = []
cat_cols = []

for i in train[cols]:
    if(train[i].dtype=='object'):
        cat_cols.append(i)
    else:
        num_cols.append(i)
        
train[cat_cols].describe(include='all')


# In[ ]:


test[cat_cols].describe(include='all')


# In[ ]:


test['Utilities'].describe(include='all')['unique']


# In[ ]:


test[i].describe(include='all')['unique']


# In[ ]:


diff_cols = [i for i in cat_cols if train[i].describe(include='all')['unique'] != test[i].describe(include='all')['unique']]
diff_cols


# In[ ]:


for i in [train, test]:
    print(i.shape)
    i.drop(diff_cols, axis=1, inplace=True)
    print(i.shape)
    print()


# In[ ]:


# for i in [i for i in null_cols if i in cat_cols]:
#     sns.catplot(x=i, y='SalePrice', kind="box", data=train)
#     plt.plot()


# In[ ]:


# for i in cat_cols:
#     plt.figure(figsize=(15, 5))
#     sns.boxplot(x=i , y='SalePrice', data=train)
#     plt.plot()


# In[ ]:


# plt.figure(figsize=(40, 40))
# sns.pairplot(train)
# plt.plot()


# In[ ]:


print("Train shape : ", train.shape, "\nTest shape : ", test.shape)


# In[ ]:


sc = StandardScaler()

for df in [train, test]:
    for i in num_cols:
        df[i] = df[i].astype('float64')
        df[i] = sc.fit_transform(df[i].values.reshape(-1,1))


# In[ ]:


train = pd.get_dummies(train, drop_first=True)
test = pd.get_dummies(test, drop_first=True)


# In[ ]:


# creating numerical and categorical columns

cols = train.drop(['Id', 'SalePrice'], axis=1).columns

num_cols = []
cat_cols = []

for i in train[cols]:
    if(train[i].dtype=='object'):
        cat_cols.append(i)
    else:
        num_cols.append(i)
        
# print(num_cols)
# print()
# print(cat_cols)


# In[ ]:


cols


# In[ ]:


train.drop(cat_cols, axis=1, inplace=True)
test.drop(cat_cols, axis=1, inplace=True)

X = train[['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold', 'MSZoning', 'LandContour', 'LotConfig', 'Neighborhood',
        'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
        'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'Foundation', 'GarageType', 'MiscFeature', 
        'SaleType', 'SaleCondition', 'Electrical', 'Heating']]
# .drop(['Id', 'SalePrice'], axis=1)
y = train['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# In[ ]:


lr = LinearRegression()
lr.fit(X_train, y_train)

pred = lr.predict(X_test)

print(mean_absolute_error(y_test, pred))
print(mean_squared_error(y_test, pred))
print(r2_score(y_test, pred))

perm = PermutationImportance(lr, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())


# In[ ]:


ridge = Ridge()
ridge.fit(X_train, y_train)

pred = ridge.predict(X_test)

print(mean_absolute_error(y_test, pred))
print(mean_squared_error(y_test, pred))
print(r2_score(y_test, pred))

perm = PermutationImportance(ridge, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())


# In[ ]:


Lasso = Lasso()
Lasso.fit(X_train, y_train)

pred = Lasso.predict(X_test)

print(mean_absolute_error(y_test, pred))
print(mean_squared_error(y_test, pred))
print(r2_score(y_test, pred))

perm = PermutationImportance(Lasso, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())


# In[ ]:


dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)

pred = dt.predict(X_test)

print(mean_absolute_error(y_test, pred))
print(mean_squared_error(y_test, pred))
print(r2_score(y_test, pred))

perm = PermutationImportance(dt, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())


# In[ ]:


rf = RandomForestRegressor()
rf.fit(X_train, y_train)

pred = rf.predict(X_test)

print(mean_absolute_error(y_test, pred))
print(mean_squared_error(y_test, pred))
print(r2_score(y_test, pred))

perm = PermutationImportance(rf, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())


# In[ ]:


# model = xgboost.XGBRegressor(colsample_bytree=0.4,
#                  gamma=0,                 
#                  learning_rate=0.07,
#                  max_depth=3,
#                  min_child_weight=1.5,
#                  n_estimators=10000,                                                                    
#                  reg_alpha=0.75,
#                  reg_lambda=0.45,
#                  subsample=0.6,
#                  seed=42) 

# model.fit(X_train, y_train)

# pred = model.predict(X_test)

# print(mean_absolute_error(y_test, pred))
# print(mean_squared_error(y_test, pred))
# print(r2_score(y_test, pred))


# In[ ]:


# feature = X_train.columns
# importance = rf.feature_importances_
# indices = np.argsort(importance)

# plt.rcParams['figure.figsize'] = (10, 50)
# plt.barh(range(len(indices)), importance[indices])
# plt.yticks(range(len(indices)), feature[indices])
# plt.xlabel('Relative Importance')
# plt.show()


# In[ ]:


# pred = rf.predict(test[['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
#         'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
#         'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
#         'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
#         'YrSold', 'MoSold', 'MSZoning', 'LandContour', 'LotConfig', 'Neighborhood',
#         'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
#         'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'Foundation', 'GarageType', 'MiscFeature', 
#         'SaleType', 'SaleCondition', 'Electrical', 'Heating']])
# my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': pred})
# my_submission.to_csv('submission.csv', index=False)

# print(my_submission.head())

