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


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


# ### Data fields
# Here's a brief version of what you'll find in the data description file.
# 
# * SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
# * MSSubClass: The building class
# * MSZoning: The general zoning classification
# * LotFrontage: Linear feet of street connected to property
# * LotArea: Lot size in square feet
# * Street: Type of road access
# * Alley: Type of alley access
# * LotShape: General shape of property
# * LandContour: Flatness of the property
# * Utilities: Type of utilities available
# * LotConfig: Lot configuration
# * LandSlope: Slope of property
# * Neighborhood: Physical locations within Ames city limits
# * Condition1: Proximity to main road or railroad
# * Condition2: Proximity to main road or railroad (if a second is present)
# * BldgType: Type of dwelling
# * HouseStyle: Style of dwelling
# * OverallQual: Overall material and finish quality
# * OverallCond: Overall condition rating
# * YearBuilt: Original construction date
# * YearRemodAdd: Remodel date
# * RoofStyle: Type of roof
# * RoofMatl: Roof material
# * Exterior1st: Exterior covering on house
# * Exterior2nd: Exterior covering on house (if more than one material)
# * MasVnrType: Masonry veneer type
# * MasVnrArea: Masonry veneer area in square feet
# * ExterQual: Exterior material quality
# * ExterCond: Present condition of the material on the exterior
# * Foundation: Type of foundation
# * BsmtQual: Height of the basement
# * BsmtCond: General condition of the basement
# * BsmtExposure: Walkout or garden level basement walls
# * BsmtFinType1: Quality of basement finished area
# * BsmtFinSF1: Type 1 finished square feet
# * BsmtFinType2: Quality of second finished area (if present)
# * BsmtFinSF2: Type 2 finished square feet
# * BsmtUnfSF: Unfinished square feet of basement area
# * TotalBsmtSF: Total square feet of basement area
# * Heating: Type of heating
# * HeatingQC: Heating quality and condition
# * CentralAir: Central air conditioning
# * Electrical: Electrical system
# * 1stFlrSF: First Floor square feet
# * 2ndFlrSF: Second floor square feet
# * LowQualFinSF: Low quality finished square feet (all floors)
# * GrLivArea: Above grade (ground) living area square feet
# * BsmtFullBath: Basement full bathrooms
# * BsmtHalfBath: Basement half bathrooms
# * FullBath: Full bathrooms above grade
# * HalfBath: Half baths above grade
# * Bedroom: Number of bedrooms above basement level
# * Kitchen: Number of kitchens
# * KitchenQual: Kitchen quality
# * TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
# * Functional: Home functionality rating
# * Fireplaces: Number of fireplaces
# * FireplaceQu: Fireplace quality
# * GarageType: Garage location
# * GarageYrBlt: Year garage was built
# * GarageFinish: Interior finish of the garage
# * GarageCars: Size of garage in car capacity
# * GarageArea: Size of garage in square feet
# * GarageQual: Garage quality
# * GarageCond: Garage condition
# * PavedDrive: Paved driveway
# * WoodDeckSF: Wood deck area in square feet
# * OpenPorchSF: Open porch area in square feet
# * EnclosedPorch: Enclosed porch area in square feet
# * 3SsnPorch: Three season porch area in square feet
# * ScreenPorch: Screen porch area in square feet
# * PoolArea: Pool area in square feet
# * PoolQC: Pool quality
# * Fence: Fence quality
# * MiscFeature: Miscellaneous feature not covered in other categories
# * MiscVal: $Value of miscellaneous feature
# * MoSold: Month Sold
# * YrSold: Year Sold
# * SaleType: Type of sale
# * SaleCondition: Condition of sale

# In[ ]:


df_train_original = pd.read_csv('../input/train.csv')
df_test_original = pd.read_csv('../input/test.csv')


# In[ ]:


df_train_original.columns


# In[ ]:


df_train_original.head()


# In[ ]:


df_test_original.head()


# ### Studing this Variables

# In[ ]:


cols = ['LotArea', 'YearBuilt', 'GarageArea', 'FullBath']
cols_test = cols + ['Id']
cols_train = cols + ['Id', 'SalePrice']


# #### Data Cleaning

# In[ ]:


df_train = df_train_original[cols_train].copy()
df_test = df_test_original[cols_test].copy()
print("Nan of trains", df_train[df_train.isnull().any(axis=1)])
print("Nan of tests", df_test[df_test.isnull().any(axis=1)])


# In[ ]:


df_test = df_test.fillna(df_test.median())
print("Nan of tests", df_test[df_test.isnull().any(axis=1)])


# In[ ]:


df_test


# > ### Descriptive

# In[ ]:


cols_with_dependent = ['SalePrice'] + cols

num_lines = len(cols_with_dependent)
fig, ax = plt.subplots(nrows=num_lines, ncols=1, figsize=(20, num_lines*5))

for index, col in enumerate(cols_with_dependent):
    ax[index].hist(df_train[col])
    ax[index].title.set_text(f"Histogram of {col}")
    ax[index].set_xlabel(f"{col}")
    ax[index].set_ylabel("Frequency")

plt.show()


# In[ ]:


boxplot = df_train.boxplot(column=['SalePrice'])


# Suggests that above $400.000,00 are outliers. Droping...

# In[ ]:


df_train_without_outliers = df_train[df_train['SalePrice'] < 400000].copy()


# In[ ]:


df_train['L_SalePrice'] = np.log(df_train['SalePrice'])
df_train_without_outliers['L_SalePrice'] = np.log(df_train_without_outliers['SalePrice'])
plt.hist(df_train['L_SalePrice'])


# In[ ]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
ax[0].boxplot(df_train['L_SalePrice'])
ax[0].title.set_text("Histogram Log")
ax[0].set_ylabel("Log Sale' Price")

ax[1].boxplot(df_train_without_outliers['L_SalePrice'])
ax[1].title.set_text("Histogram Log Without outliers")
ax[1].set_ylabel("Log Sale' Price")


# In[ ]:


# plt.plot(df_train_original['YearBuilt'], df_train_original['SalePrice'],  'o')
# plt.title("Year Build X Sale's Price")
# plt.xlabel('Year Build')
# plt.ylabel("Sale's Price")

num_lines = len(cols)
fig, ax = plt.subplots(nrows=num_lines, ncols=2, figsize=(20, num_lines * num_lines))

for index, col in enumerate(cols):
    ax[index, 0].scatter(df_train[col], df_train['SalePrice'])
    ax[index, 0].title.set_text(f"{col} X Sale's Price")
    ax[index, 0].set_xlabel(f"{col}")
    ax[index, 0].set_ylabel("Sale's Price")
    
    ax[index, 1].scatter(df_train[col], df_train['L_SalePrice'])
    ax[index, 1].title.set_text(f"{col} X Log Sale's Price")
    ax[index, 1].set_xlabel(f"{col}")
    ax[index, 1].set_ylabel("Log Sale's Price")

plt.show()


# ### Predictive Analisys

# In[ ]:


y = df_train['SalePrice']
X = df_train[cols]
reg = LinearRegression().fit(X, y)
X_test = df_test[cols].copy()
print("Score", reg.score(X, y))
y_hat = reg.predict(X_test)


# In[ ]:


# y = df_train_without_outliers['L_SalePrice']
# X = df_train_without_outliers[cols]
# reg = LinearRegression().fit(X, y)
# X_test = df_test[cols]
# print("Score", reg.score(X, y))
# reg.predict(X_test)


# ### Submitting

# In[ ]:


my_submission = pd.DataFrame({'Id': df_test.Id, 'SalePrice': y_hat})
my_submission.to_csv('submission.csv', index=False)

