#!/usr/bin/env python
# coding: utf-8

# ## Problem Description:
# Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.
# With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, **this competition challenges to predict the final price of each home.**
# 
# ## About Data:
# * train.csv - the training set
# * test.csv - the test set
# * data_description.txt - full description of each column, originally prepared by Dean De Cock but lightly edited to match the column names used here
# * sample_submission.csv - a benchmark submission from a linear regression on year and month of sale, lot square footage, and number of bedrooms
# 
# ## Data Description:
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
# * MiscVal: Value of miscellaneous feature
# * MoSold: Month Sold
# * YrSold: Year Sold
# * SaleType: Type of sale
# * SaleCondition: Condition of sale
# 
# ## Note:
# * While reading the data set we have considered 'NaN' as NAs.
# 
# ## Evaluation Metric:
# * The evaluation metric is **Root-Mean-Squared-Error (RMSE)**

# In[ ]:


import pandas as pd
import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv',header=0)
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', header = 0)
sample_submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv',header = 0)


# In[ ]:


pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


sample_submission.head()


# In[ ]:


print(train.shape)
print(test.shape)
print(sample_submission.shape)


# In[ ]:


train['SalePrice'].describe()


# In[ ]:


#histogram
sns.distplot(train['SalePrice']);


# * Deviate from the normal distribution.
# * Have appreciable positive skewness.
# * Show peakedness.

# ### Relationship with numerical variables

# In[ ]:


#scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# It seems that 'SalePrice' and 'GrLivArea' are in linear relationship.

# In[ ]:


#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# 'TotalBsmtSF' and 'SalePrice' are also in Liner relation

# ### Relationship with categorical features

# In[ ]:


#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# In[ ]:


var = 'YearBuilt'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);


# In[ ]:


sns.set(font_scale=1)
correlation_train=train.corr()
plt.figure(figsize=(30,20))
sns.heatmap(correlation_train,annot=True,fmt='.1f')


# In[ ]:


train.corr()


# ### Scatter plots between 'SalePrice' and correlated variables

# In[ ]:


#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size = 2.5)
plt.show();


# ## Missing Data

# In[ ]:


#missing data
train_total = train.isnull().sum().sort_values(ascending=False)
train_percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
train_missing_data = pd.concat([train_total, train_percent], axis=1, keys=['Total', 'Percent'])
train_missing_data.head(20)


# In[ ]:


#missing data in test
test_total = test.isnull().sum().sort_values(ascending=False)
test_percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)
test_missing_data = pd.concat([test_total, test_percent], axis=1, keys=['Total', 'Percent'])
test_missing_data.head(35)


# ## Imputing missing values
# We impute them by proceeding sequentially through features with missing values
# * **PoolQC** : data description says NA means "No Pool". That make sense, given the huge ratio of missing value (+99%) and majority of houses have no Pool at all in general.
# * **MiscFeature** : data description says NA means "no misc feature".
# * **Alley** : data description says NA means "no alley access".
# * **Fence** : data description says NA means "no fence".
# * **FireplaceQu** : data description says NA means "no fireplace".
# * **GarageType, GarageFinish, GarageQual and GarageCond** : Replacing missing data with None
# * **LotFrontage** : Since the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood , we can **fill in missing values by the median LotFrontage of the neighborhood.**
# * **GarageYrBlt, GarageArea and GarageCars** : Replacing missing data with 0 (Since No garage = no cars in such garage.)
# * **BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath** : missing values are likely zero for having no basement.
# * **BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2** : For all these categorical basement-related features, NaN means that there is no basement.
# * **MasVnrArea and MasVnrType** : NA most likely means no masonry veneer for these houses. We can fill 0 for the area and None for the type.
# * **MSZoning (The general zoning classification)** : 'RL' is by far the most common value. So we can fill in missing values with 'RL'.
# * **Functional** : data description says NA means typical.
# * **Electrical** : It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.
# * **KitchenQual**: Only one NA value, and same as Electrical, we set 'TA' (which is the most frequent) for the missing value in KitchenQual.
# * **Exterior1st and Exterior2nd** : Again Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string.
# * **SaleType** : Fill in again with most frequent which is "WD".
# * **MSSubClass** : Na most likely means No building class. We can replace missing values with None
# 
# 
# * **Utilities** : For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . Since the house with 'NoSewa' is in the training set, **this feature won't help in predictive modelling**. We can then safely remove it.

# In[ ]:


# Train Data Imputation

for col in ('PoolQC','MiscFeature','Alley','Fence','GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','MasVnrType','MSSubClass','FireplaceQu'):
    train[col] = train[col].fillna('None')
    

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF',
            'BsmtFullBath', 'BsmtHalfBath','MasVnrArea'):
    train[col] = train[col].fillna(0)
    

train["LotFrontage"] = train.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

train["Functional"] = train["Functional"].fillna("Typ")

train['MSZoning'] = train['MSZoning'].fillna(train['MSZoning'].mode()[0])
train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])
train['KitchenQual'] = train['KitchenQual'].fillna(train['KitchenQual'].mode()[0])
train['Exterior1st'] = train['Exterior1st'].fillna(train['Exterior1st'].mode()[0])
train['Exterior2nd'] = train['Exterior2nd'].fillna(train['Exterior2nd'].mode()[0])
train['SaleType'] = train['SaleType'].fillna(train['SaleType'].mode()[0])


train = train.drop(['Utilities'], axis=1)


# In[ ]:


train_total = train.isnull().sum().sort_values(ascending=False)
train_percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
train_missing_data = pd.concat([train_total, train_percent], axis=1, keys=['Total', 'Percent'])
train_missing_data.head(5)


# In[ ]:


# Test Data Imputation

for col in ('PoolQC','MiscFeature','Alley','Fence','GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','MasVnrType','MSSubClass','FireplaceQu'):
    test[col] = test[col].fillna('None')
    

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF',
            'BsmtFullBath', 'BsmtHalfBath','MasVnrArea'):
    test[col] = test[col].fillna(0)
    

test["LotFrontage"] = test.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

test["Functional"] = test["Functional"].fillna("Typ")

test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode()[0])
test['Electrical'] = test['Electrical'].fillna(test['Electrical'].mode()[0])
test['KitchenQual'] = test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])
test['Exterior1st'] = test['Exterior1st'].fillna(test['Exterior1st'].mode()[0])
test['Exterior2nd'] = test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0])
test['SaleType'] = test['SaleType'].fillna(test['SaleType'].mode()[0])


test = test.drop(['Utilities'], axis=1)


# In[ ]:


#missing data in test
test_total = test.isnull().sum().sort_values(ascending=False)
test_percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)
test_missing_data = pd.concat([test_total, test_percent], axis=1, keys=['Total', 'Percent'])
test_missing_data.head(5)


# In[ ]:


train.head(5)


# In[ ]:


test.head(5)


# In[ ]:


# Print the number of unique levels in train and test data
train_unique = train.nunique().sort_values(ascending=False)
test_unique = test.nunique().sort_values(ascending=False)
unique_data = pd.concat([train_unique, test_unique], axis=1, keys=['Train', 'Test'],join="inner")
unique_data.head(100)


# ### Converting the data into 'int' and 'category'

# In[ ]:


# Train data
train['LotFrontage'] = train['LotFrontage'].astype('int64')
train['MasVnrArea'] = train['MasVnrArea'].astype('int64')
train['GarageYrBlt'] = train['GarageYrBlt'].astype('int64')
# Test Data
test['LotFrontage'] = test['LotFrontage'].astype('int64')
test['MasVnrArea'] = test['MasVnrArea'].astype('int64')
test['GarageYrBlt'] = test['GarageYrBlt'].astype('int64')
test['BsmtFinSF1'] = test['BsmtFinSF1'].astype('int64')
test['BsmtFinSF2'] = test['BsmtFinSF2'].astype('int64')
test['BsmtUnfSF'] = test['BsmtUnfSF'].astype('int64')
test['TotalBsmtSF'] = test['TotalBsmtSF'].astype('int64')
test['GarageArea'] = test['GarageArea'].astype('int64')
test['BsmtFullBath'] = test['BsmtFullBath'].astype('int64')
test['BsmtHalfBath'] = test['BsmtHalfBath'].astype('int64')


# In[ ]:


for col in ('Alley','BldgType','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtQual',
            'CentralAir','Condition1','ExterCond','ExterQual','Fence','FireplaceQu','Foundation',
            'Functional','GarageCond','GarageFinish','GarageType','HeatingQC','KitchenQual',
            'LandContour','LandSlope','LotConfig','LotShape','MasVnrType','MSZoning',
            'Neighborhood','PavedDrive','RoofStyle','SaleCondition','SaleType','Street',
           'OverallQual','OverallCond','BsmtFullBath','BsmtHalfBath','HalfBath','TotRmsAbvGrd',
           'MoSold','YrSold') :
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')


# ### Drop the columns because the unique levels are different with train and test

# In[ ]:


# Train data
train.drop(['Condition2','Electrical','Exterior1st','Exterior2nd','GarageQual',
            'Heating','HouseStyle','MiscFeature','PoolQC','RoofMatl',
           'MSSubClass','LowQualFinSF','FullBath','BedroomAbvGr','KitchenAbvGr',
            'Fireplaces','GarageCars','3SsnPorch','PoolArea','MiscVal','Id'], axis=1, inplace=True)
# Test data
test.drop(['Condition2','Electrical','Exterior1st','Exterior2nd','GarageQual',
            'Heating','HouseStyle','MiscFeature','PoolQC','RoofMatl',
           'MSSubClass','LowQualFinSF','FullBath','BedroomAbvGr','KitchenAbvGr',
            'Fireplaces','GarageCars','3SsnPorch','PoolArea','MiscVal','Id'], axis=1, inplace=True)


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.head()


# In[ ]:


test.head()


# ### Select Categorical Feature Indices

# In[ ]:


categorical_features_indices = np.where(train.dtypes != np.int64)[0]


# In[ ]:


categorical_features_indices


# ## Split the data

# In[ ]:


from sklearn.model_selection import train_test_split

y = train["SalePrice"]
X = train.drop('SalePrice', axis=1)
X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.75, random_state=789)


# ## Model - CatBoostRegressor

# In[ ]:


from catboost import CatBoostRegressor


# In[ ]:


model_1=CatBoostRegressor(loss_function='RMSE')
model_1.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_validation, y_validation),plot=True)


# In[ ]:


pred =model_1.predict(test)


# In[ ]:


submission = pd.DataFrame()
submission['Id'] = sample_submission.Id
submission['SalePrice'] = pred
submission.to_csv('submission.csv', index=False)
submission.head(5)


# In[ ]:




