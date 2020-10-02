#!/usr/bin/env python
# coding: utf-8

# ## Problem Statement

# Predicting the house prices using the features in the data given

# ### Importing libraries

# In[ ]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns


# ### Reading the data

# In[ ]:


df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')


# In[ ]:


df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


df_s = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')


# ### EDA

# In[ ]:


df_train.head()


# #### Data description:
# 
# - MSSubClass: Identifies the type of dwelling involved in the sale.
# - MSZoning: Identifies the general zoning classification of the sale.
# - LotFrontage: Linear feet of street connected to property.
# - LotArea: Lot size in square feet.
# - Street: Type of road access to property.
# - Alley: Type of alley access to property.
# - LotShape: General shape of property.
# - LandContour: Flatness of the property.
# - Utilities: Type of utilities available.
# - LotConfig: Lot configuration.
# - LandSlope: Slope of property.
# - Neighborhood: Physical locations within Ames city limits.
# - Condition1: Proximity to various conditions.
# - Condition2: Proximity to various conditions (if more than one is present).
# - BldgType: Type of dwelling.
# - HouseStyle: Style of dwelling.
# - OverallQual: Rates the overall material and finish of the house.
# - OverallCond: Rates the overall condition of the house.
# - YearBuilt: Original construction date.
# - YearRemodAdd: Remodel date (same as construction date if no remodeling or additions).
# - RoofStyle: Type of roof.
# - RoofMatl: Roof material.
# - Exterior1st: Exterior covering on house.
# - Exterior2nd: Exterior covering on house (if more than one material).
# - MasVnrType: Masonry veneer type (None - None).
# - MasVnrArea: Masonry veneer area in square feet.
# - ExterQual: Evaluates the quality of the material on the exterior/
# - ExterCond: Evaluates the present condition of the material on the exterior.
# - Foundation: Type of foundation.
# - BsmtQual: Evaluates the height of the basement (NA - No Basement).
# - BsmtCond: Evaluates the general condition of the basement (NA - No Basement).
# - BsmtExposure: Refers to walkout or garden level walls (NA - No Basement).
# - BsmtFinType1: Rating of basement finished area (NA - No Basement).
# - BsmtFinSF1: Type 1 finished square feet.
# - BsmtFinType2: Rating of basement finished area (if multiple types) (NA - No Basement).
# - BsmtFinSF2: Type 2 finished square feet.
# - BsmtUnfSF: Unfinished square feet of basement area.
# - TotalBsmtSF: Total square feet of basement area.
# - Heating: Type of heating.
# - HeatingQC: Heating quality and condition.
# - CentralAir: Central air conditioning.
# - Electrical: Electrical system.
# - 1stFlrSF: First Floor square feet.
# - 2ndFlrSF: Second floor square feet.
# - LowQualFinSF: Low quality finished square feet (all floors).
# - GrLivArea: Above grade (ground) living area square feet.
# - BsmtFullBath: Basement full bathrooms.
# - BsmtHalfBath: Basement half bathrooms.
# - FullBath: Full bathrooms above grade.
# - HalfBath: Half baths above grade.
# - Bedroom: Bedrooms above grade (does NOT include basement bedrooms).
# - Kitchen: Kitchens above grade.
# - KitchenQual: Kitchen quality.
# - TotRmsAbvGrd: Total rooms above grade (does not include bathrooms).
# - Functional: Home functionality (Assume typical unless deductions are warranted).
# - Fireplaces: Number of fireplaces.
# - FireplaceQu: Fireplace quality (NA - No Fireplace).
# - GarageType: Garage location (NA - No Garage).
# - GarageYrBlt: Year garage was built.
# - GarageFinish: Interior finish of the garage (NA - No Garage).
# - GarageCars: Size of garage in car capacity.
# - GarageArea: Size of garage in square feet.
# - GarageQual: Garage quality (NA - No Garage).
# - GarageCond: Garage condition (NA - No Garage).
# - PavedDrive: Paved driveway.
# - WoodDeckSF: Wood deck area in square feet.
# - OpenPorchSF: Open porch area in square feet.
# - EnclosedPorch: Enclosed porch area in square feet.
# - 3SsnPorch: Three season porch area in square feet.
# - ScreenPorch: Screen porch area in square feet.
# - PoolArea: Pool area in square feet.
# - PoolQC: Pool quality (NA - No Pool).
# - Fence: Fence quality (NA - No Fence).
# - MiscFeature: Miscellaneous feature not covered in other categories (NA - None).
# - MiscVal: Value of miscellaneous feature.
# - MoSold: Month Sold (MM).
# - YrSold: Year Sold (YYYY).
# - SaleType: Type of sale.
# - SaleCondition: Condition of sale.

# In[ ]:


df_test.head()


# In[ ]:


df_s.head()


# In[ ]:


df_train.shape


# In[ ]:


df_train.nunique()


# ##### Checking Null Values

# In[ ]:


null_values = df_train.isnull().sum()
null_values


# In[ ]:


n_v = null_values[null_values>0]
n_v = n_v/df_train.shape[0]*100
n_v


# In[ ]:


n_v20 = (n_v[n_v>20])
n_v20


# ###### Droping the columns which is having more than 20% of null values in the data

# In[ ]:


df_train.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'], axis=1, inplace=True)
df_test.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'], axis=1, inplace=True)


# In[ ]:


df_train.head()


# #### Visualization

# In[ ]:


sns.distplot(df_train['SalePrice'])


# In[ ]:


plt.figure(figsize = (12, 6))
plt.subplot(121)
plot1 = plt.scatter(range(df_train.shape[0]), np.sort(df_train.SalePrice.values))
plot1 = plt.title("SalePrice Curve Distribuition", fontsize=15)
plot1 = plt.xlabel("")
plot1 = plt.ylabel("SalePrice", fontsize=12)

plt.subplots_adjust(wspace = 0.3, hspace = 0.5,top = 0.9)
plt.show()


# In[ ]:


var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))


# #### Checking Correlation

# In[ ]:


df_train.corr()
plt.figure(figsize=(5,20))
sns.heatmap(df_train[['SalePrice']].sort_values(by=['SalePrice'],ascending=False).head(60),vmin=-1, annot=True)


# In[ ]:


sns.set(font_scale=1)
correlation_train=df_train.corr()
plt.figure(figsize=(30,20))
sns.heatmap(correlation_train,annot=True,fmt='.1f')


# In[ ]:


df_train.corr()


# Droping the Categorical variables

# In[ ]:


df_train.drop(['Id', 'MSZoning', 'Street','LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
             'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle','RoofStyle','RoofMatl',
             'Exterior1st', 'Exterior2nd', 'MasVnrType','ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
             'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','Heating', 'HeatingQC','CentralAir', 
             'Electrical', 'KitchenQual', 'Functional', 'GarageType','GarageFinish','GarageQual', 'GarageCond',
             'PavedDrive','SaleType','SaleCondition'], axis=1, inplace=True)


# In[ ]:


df_train.head()


# In[ ]:


df_train.columns


# In[ ]:


null_values_train = df_train.isnull().sum()
null_values_train


# #### Imputation

# In[ ]:


df_train["LotFrontage"] = df_train['LotFrontage'].transform(lambda x: x.fillna(x.mean()))


# In[ ]:


df_train["MasVnrArea"] = df_train['MasVnrArea'].transform(lambda x: x.fillna(x.mean()))


# In[ ]:


df_train["GarageYrBlt"] = df_train['GarageYrBlt'].transform(lambda x: x.fillna(x.median()))


# In[ ]:


df_train.isnull().sum()


# In[ ]:


Id = df_test['Id']


# In[ ]:


df_test.drop(['Id', 'MSZoning', 'Street','LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
             'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle','RoofStyle','RoofMatl',
             'Exterior1st', 'Exterior2nd', 'MasVnrType','ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
             'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','Heating', 'HeatingQC','CentralAir', 
             'Electrical', 'KitchenQual', 'Functional', 'GarageType','GarageFinish','GarageQual', 'GarageCond',
             'PavedDrive','SaleType','SaleCondition'], axis=1, inplace=True)


# In[ ]:


df_test.head()


# In[ ]:


df_test.columns


# In[ ]:


df_test.isnull().sum()


# In[ ]:


df_test["LotFrontage"] = df_test['LotFrontage'].transform(lambda x: x.fillna(x.mean()))
df_test["MasVnrArea"] = df_test['MasVnrArea'].transform(lambda x: x.fillna(x.mean()))
df_test["GarageYrBlt"] = df_test['GarageYrBlt'].transform(lambda x: x.fillna(x.median()))
df_test["BsmtFinSF1"] = df_test['BsmtFinSF1'].transform(lambda x: x.fillna(x.mean()))
df_test["BsmtFinSF2"] = df_test['BsmtFinSF2'].transform(lambda x: x.fillna(x.mean()))
df_test["BsmtUnfSF"] = df_test['BsmtUnfSF'].transform(lambda x: x.fillna(x.mean()))
df_test["TotalBsmtSF"] = df_test['TotalBsmtSF'].transform(lambda x: x.fillna(x.mean()))
df_test["BsmtFullBath"] = df_test['BsmtFullBath'].transform(lambda x: x.fillna(x.mean()))
df_test["BsmtHalfBath"] = df_test['BsmtHalfBath'].transform(lambda x: x.fillna(x.mean()))
df_test["GarageCars"] = df_test['GarageCars'].transform(lambda x: x.fillna(x.mean()))
df_test["GarageArea"] = df_test['GarageArea'].transform(lambda x: x.fillna(x.mean()))


# In[ ]:


df_test.isnull().sum()


# ### Model Building

# In[ ]:


from sklearn.model_selection import cross_val_score, train_test_split 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error as MSE


# In[ ]:


X = df_train.drop(["SalePrice"],axis=1).values
y = df_train["SalePrice"].values


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=116214)


# #### Linear Regression

# In[ ]:


lr = LinearRegression()
lr.fit(X_train,y_train)


# In[ ]:


print(f"Train score : {lr.score(X_train,y_train)}")
print(f"Validation score : {lr.score(X_test,y_test)}")


# In[ ]:


Prediction_LR = lr.predict(df_test)


# In[ ]:


submission_LR = pd.DataFrame()
submission_LR['Id'] = Id
submission_LR['SalePrice'] = Prediction_LR
submission_LR.to_csv('submission_LR.csv', index=False)
submission_LR.head(5)


# #### SVM

# In[ ]:


SVM = SVR()
SVM.fit(X_train,y_train)


# In[ ]:


print(f"Train score : {SVM.score(X_train,y_train)}")
print(f"Validation score : {SVM.score(X_test,y_test)}")


# In[ ]:


Prediction_SVM = SVM.predict(df_test)


# In[ ]:


submission_SVM = pd.DataFrame()
submission_SVM['Id'] = Id
submission_SVM['SalePrice'] = Prediction_SVM
submission_SVM.to_csv('submission_SVM.csv', index=False)
submission_SVM.head(5)


# #### DT

# In[ ]:


DT = DecisionTreeRegressor()
DT.fit(X_train,y_train)


# In[ ]:


print(f"Train score : {DT.score(X_train,y_train)}")
print(f"Validation score : {DT.score(X_test,y_test)}")


# In[ ]:


Prediction_DT = DT.predict(df_test)


# In[ ]:


submission_DT = pd.DataFrame()
submission_DT['Id'] = Id
submission_DT['SalePrice'] = Prediction_DT
submission_DT.to_csv('submission_DT.csv', index=False)
submission_DT.head(5)


# #### KNN

# In[ ]:


KNN = KNeighborsRegressor()
KNN.fit(X_train,y_train)


# In[ ]:


print(f"Train score : {KNN.score(X_train,y_train)}")
print(f"Validation score : {KNN.score(X_test,y_test)}")


# In[ ]:


Prediction_KNN = KNN.predict(df_test)


# In[ ]:


submission_KNN = pd.DataFrame()
submission_KNN['Id'] = Id
submission_KNN['SalePrice'] = Prediction_KNN
submission_KNN.to_csv('submission_KNN.csv', index=False)
submission_KNN.head(5)


# #### Random Forest

# In[ ]:


RF = RandomForestRegressor()
RF.fit(X_train,y_train)


# In[ ]:


print(f"Train score : {RF.score(X_train,y_train)}")
print(f"Validation score : {RF.score(X_test,y_test)}")


# In[ ]:


Prediction_RF = RF.predict(df_test)


# In[ ]:


submission_RF = pd.DataFrame()
submission_RF['Id'] = Id
submission_RF['SalePrice'] = Prediction_RF
submission_RF.to_csv('submission_RF.csv', index=False)
submission_RF.head(5)


# In[ ]:


param_grid = {'bootstrap': [True, False],
               'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
               'max_features': ['auto', 'sqrt'],
               'min_samples_leaf': [1, 2, 4],
               'min_samples_split': [2, 5, 10],
               'n_estimators': [130, 180, 230]}


# In[ ]:


RF_random = RandomizedSearchCV(estimator = RF, param_distributions = param_grid, n_iter = 100, cv = 10, verbose=2, random_state=42, n_jobs = -1)


# In[ ]:


RF_random.fit(X_train,y_train)


# In[ ]:


print(f"Train score : {RF_random.score(X_train,y_train)}")
print(f"Validation score : {RF_random.score(X_test,y_test)}")


# In[ ]:


Prediction_RF1 = RF_random.predict(df_test)


# In[ ]:


submission_RF1 = pd.DataFrame()
submission_RF1['Id'] = Id
submission_RF1['SalePrice'] = Prediction_RF1
submission_RF1.to_csv('submission_RF1.csv', index=False)
submission_RF1.head(5)


# In[ ]:


param_grid_2 = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8]
}


# In[ ]:


RF_grid = GridSearchCV(estimator = RF, param_grid = param_grid_2, cv = 10)


# In[ ]:


RF_grid.fit(X_train,y_train)


# In[ ]:


print(f"Train score : {RF_grid.score(X_train,y_train)}")
print(f"Validation score : {RF_grid.score(X_test,y_test)}")


# In[ ]:


Prediction_RF2 = RF_grid.predict(df_test)


# In[ ]:


submission_RF2 = pd.DataFrame()
submission_RF2['Id'] = Id
submission_RF2['SalePrice'] = Prediction_RF2
submission_RF2.to_csv('submission_RF2.csv', index=False)
submission_RF2.head(5)


# In[ ]:


from catboost import Pool, CatBoostClassifier


# In[ ]:


CBR = CatBoostClassifier(iterations=100)


# In[ ]:


CBR.fit(X_train, y_train)


# In[ ]:


print('Accuracy of classifier on training set: {:.2f}'.format(CBR.score(X_train, y_train) * 100))
print('Accuracy of classifier on test set: {:.2f}'.format(CBR.score(X_test, y_test) * 100))


# In[ ]:


prediction_CBR = CBR.predict(df_test)


# In[ ]:


submission_CBR = pd.DataFrame()
submission_CBR['Id'] = Id
submission_CBR['SalePrice'] = prediction_CBR
submission_CBR.to_csv('submission_CBR.csv', index=False)
submission_CBR.head(5)


# In[ ]:




