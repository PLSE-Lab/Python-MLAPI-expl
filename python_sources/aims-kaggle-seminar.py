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


# ---
# # Load data

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[ ]:


train = pd.read_csv('../input/train.csv')
print(train.shape)
train.head()


# In[ ]:


test = pd.read_csv('../input/test.csv')
print(test.shape)
test.head()


# In[ ]:


sample_sub = pd.read_csv('../input/sample_submission.csv')
print(sample_sub.shape)
sample_sub.head()


# In[ ]:


df = pd.concat([train, test], sort=False).reset_index(drop=True)
print(df.shape)
df.head()


# In[ ]:


df.tail()


# In[ ]:


features = df.columns[1:-1]
print(len(features))
features


# In[ ]:


num_features = train.select_dtypes(include='number').columns[1:-1]
cat_features = train.select_dtypes(exclude='number').columns


# ---
# # Data fields
# Here's a brief version of what you'll find in the data description file.
# 
# - **SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.**
# - MSSubClass: The building class
# - MSZoning: The general zoning classification
# - LotFrontage: Linear feet of street connected to property
# - LotArea: Lot size in square feet
# - Street: Type of road access
# - Alley: Type of alley access
# - LotShape: General shape of property
# - LandContour: Flatness of the property
# - Utilities: Type of utilities available
# - LotConfig: Lot configuration
# - LandSlope: Slope of property
# - Neighborhood: Physical locations within Ames city limits
# - Condition1: Proximity to main road or railroad
# - Condition2: Proximity to main road or railroad (if a second is present)
# - BldgType: Type of dwelling
# - HouseStyle: Style of dwelling
# - OverallQual: Overall material and finish quality
# - OverallCond: Overall condition rating
# - YearBuilt: Original construction date
# - YearRemodAdd: Remodel date
# - RoofStyle: Type of roof
# - RoofMatl: Roof material
# - Exterior1st: Exterior covering on house
# - Exterior2nd: Exterior covering on house (if more than one material)
# - MasVnrType: Masonry veneer type
# - MasVnrArea: Masonry veneer area in square feet
# - ExterQual: Exterior material quality
# - ExterCond: Present condition of the material on the exterior
# - Foundation: Type of foundation
# - BsmtQual: Height of the basement
# - BsmtCond: General condition of the basement
# - BsmtExposure: Walkout or garden level basement walls
# - BsmtFinType1: Quality of basement finished area
# - BsmtFinSF1: Type 1 finished square feet
# - BsmtFinType2: Quality of second finished area (if present)
# - BsmtFinSF2: Type 2 finished square feet
# - BsmtUnfSF: Unfinished square feet of basement area
# - TotalBsmtSF: Total square feet of basement area
# - Heating: Type of heating
# - HeatingQC: Heating quality and condition
# - CentralAir: Central air conditioning
# - Electrical: Electrical system
# - 1stFlrSF: First Floor square feet
# - 2ndFlrSF: Second floor square feet
# - LowQualFinSF: Low quality finished square feet (all floors)
# - GrLivArea: Above grade (ground) living area square feet
# - BsmtFullBath: Basement full bathrooms
# - BsmtHalfBath: Basement half bathrooms
# - FullBath: Full bathrooms above grade
# - HalfBath: Half baths above grade
# - Bedroom: Number of bedrooms above basement level
# - Kitchen: Number of kitchens
# - KitchenQual: Kitchen quality
# - TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
# - Functional: Home functionality rating
# - Fireplaces: Number of fireplaces
# - FireplaceQu: Fireplace quality
# - GarageType: Garage location
# - GarageYrBlt: Year garage was built
# - GarageFinish: Interior finish of the garage
# - GarageCars: Size of garage in car capacity
# - GarageArea: Size of garage in square feet
# - GarageQual: Garage quality
# - GarageCond: Garage condition
# - PavedDrive: Paved driveway
# - WoodDeckSF: Wood deck area in square feet
# - OpenPorchSF: Open porch area in square feet
# - EnclosedPorch: Enclosed porch area in square feet
# - 3SsnPorch: Three season porch area in square feet
# - ScreenPorch: Screen porch area in square feet
# - PoolArea: Pool area in square feet
# - PoolQC: Pool quality
# - Fence: Fence quality
# - MiscFeature: Miscellaneous feature not covered in other categories
# - MiscVal: $Value of miscellaneous feature
# - MoSold: Month Sold
# - YrSold: Year Sold
# - SaleType: Type of sale
# - SaleCondition: Condition of sale

# In[ ]:


import pandas_profiling


# In[ ]:


pandas_profiling.ProfileReport(df)


# ---
# # Distribution of target

# In[ ]:


target = train['SalePrice']
target.head(10)


# In[ ]:


target.describe()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=[20, 10])
target.hist(bins=100)


# ---
# # Correlation of features

# In[ ]:


corr_mat = train.loc[:, num_features].corr()
plt.figure(figsize=[15, 15])
sns.heatmap(corr_mat, square=True)


# ---
# # Correlation of feature and target

# In[ ]:


fig = plt.figure(figsize=[30, 30])
plt.tight_layout()

for i, feature in enumerate(num_features):
    ax = fig.add_subplot(6, 6, i+1)
    sns.regplot(x=train.loc[:, feature],
                y=train.loc[:, 'SalePrice'])


# In[ ]:


fig = plt.figure(figsize=[30, 40])
plt.tight_layout()

for i, feature in enumerate(cat_features):
    ax = fig.add_subplot(9, 5, i+1)
    sns.violinplot(x=df.loc[:, feature],
                   y=df.loc[:, 'SalePrice'])


# ---
# # Label Encoding

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[ ]:


for col in cat_features:
    df[col] = df[col].fillna('NULL')
    df[col+'_le'] = le.fit_transform(df[col])


# In[ ]:


df = df.drop(cat_features, axis=1)


# In[ ]:


df.head()


# In[ ]:


le_features = []
for feat in cat_features:
    le_features.append(feat+'_le')


# In[ ]:


len(le_features)


# ---
# # Fill NaN

# In[ ]:


for feat in num_features:
    df[feat] = df[feat].fillna(-1)


# ---
# # Data split

# In[ ]:


train = df[df['Id'].isin(train['Id'])]
test = df[df['Id'].isin(test['Id'])]


# In[ ]:


X_train = train.drop(['Id', 'SalePrice'], axis=1)
y_train = train['SalePrice']

X_test = test.drop(['Id', 'SalePrice'], axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train_, X_val, y_train_, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# ---
# # Hold-out validation
# ## Model: Ridge Regression

# In[ ]:


from sklearn.linear_model import Ridge


# In[ ]:


reg = Ridge(alpha=0.3, random_state=42)


# In[ ]:


reg.fit(X_train_, y_train_)


# In[ ]:


from sklearn.metrics import mean_squared_error
def metric(y_true, y_pred):
    return mean_squared_error(np.log(y_true), np.log(y_pred)) ** 0.5


# In[ ]:


pred_train = reg.predict(X_train_)
rmse_train = mean_squared_error(np.log(y_train_), np.log(pred_train))**0.5
rmse_train


# In[ ]:


pred_train[:5]


# In[ ]:


y_train_.head()


# In[ ]:


pred_val = reg.predict(X_val)
rmse_val = mean_squared_error(np.log(y_val), np.log(pred_val))**0.5
rmse_val


# In[ ]:


pred_test = reg.predict(X_test)
print(pred_test.shape)
pred_test[:5]


# ---
# # Submission

# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
print(sub.shape)
sub.head()


# In[ ]:


sub['SalePrice'] = pred_test
sub.head()


# In[ ]:


sub.to_csv('submission_ridge_regression.csv', index=False)


# ---
# # 5-fold CV
# ## Model: Ridge Regression

# In[41]:


from sklearn.model_selection import KFold


# In[43]:


def cv(reg, X_train, y_train, X_test):
    kf = KFold(n_splits=5, random_state=42)
    pred_test_mean = np.zeros(sub['SalePrice'].shape)
    for train_index, val_index in kf.split(X_train):
        X_train_train = X_train.iloc[train_index]
        y_train_train = y_train.iloc[train_index]

        X_train_val = X_train.iloc[val_index]
        y_train_val = y_train.iloc[val_index]

        reg.fit(X_train_train, y_train_train)
        pred_train = reg.predict(X_train_train)
        metric_train = metric(y_train_train, pred_train)
        print('train metric: ', metric_train)

        pred_val = reg.predict(X_train_val)
        metric_val = metric(y_train_val, pred_val)
        print('val metric:   ', metric_val)
        print()

        pred_test = reg.predict(X_test)
        pred_test_mean += pred_test / kf.get_n_splits()
        
    return pred_test_mean


# In[44]:


reg = Ridge(alpha=0.3, random_state=42)
pred_test_mean = cv(reg, X_train, y_train, X_test)


# In[45]:


sub['SalePrice'] = pred_test_mean
sub.head()


# In[46]:


sub.to_csv('submission_ridge_regression_5f_CV.csv', index=False)


# ---
# # Target scaling

# In[ ]:


y_train_log = np.log(y_train)
plt.figure(figsize=[20, 10])
plt.hist(y_train_log, bins=50);


# In[ ]:


reg = Ridge(alpha=0.3, random_state=42)
pred_test_mean = cv(reg, X_train, y_train_log, X_test)


# In[ ]:


sub['SalePrice'] = np.exp(pred_test_mean)
sub.to_csv('submission_ridge_regression_cv_target_log.csv', index=False)
sub.head()


# # Features scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
X_train_scaled.head()


# In[ ]:


X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test))
X_test_scaled.head()


# In[ ]:


reg = Ridge(alpha=0.3, random_state=42)
pred_test = cv(reg, X_train_scaled, y_train_log, X_test_scaled)


# In[ ]:


sub['SalePrice'] = np.exp(pred_test)
sub.to_csv('submission_ridge_regression_cv_target_log_scaled_feature.csv', index=False)
sub.head()


# ## Model: Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


reg = RandomForestRegressor(n_estimators=1000, random_state=42)
pred_test = cv(reg, X_train, y_train_log, X_test)


# In[ ]:


sub['SalePrice'] = np.exp(pred_test)
sub.to_csv('submission_random_forest_cv_target_log.csv', index=False)
sub.head()


# ---
# # Plot feature importances

# In[ ]:


reg.fit(X_train, y_train_log)


# In[ ]:


feature_importances = reg.feature_importances_
feature_importances


# In[ ]:


feature_importances = pd.DataFrame([X_train.columns, feature_importances]).T
feature_importances = feature_importances.sort_values(by=1, ascending=False)


# In[ ]:


plt.figure(figsize=[20, 20])
sns.barplot(x=feature_importances.iloc[:, 1],
            y=feature_importances.iloc[:, 0], orient='h')
plt.tight_layout()
plt.show()

