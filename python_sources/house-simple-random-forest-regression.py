#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
from sklearn.exceptions import DataConversionWarning
# Suppress warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)


# * [Load Data and Libraries](#load)
# * [Check Data](#check-data)
# * [Data Pre-Processing](#pre-processing)
# * [Training and Prediction](#training-prediction)

# # Load Data and Libraries <a id="load"></a>

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import os
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler

# Set pandas data display option
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 500)

# Display all filenames
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Load csv data
train = pd.read_csv("../input/train.csv")
compe = pd.read_csv("../input/test.csv")
sample_sub = pd.read_csv("../input/sample_submission.csv")

# All data
data  = train.append(compe, sort=False)


# # Check Data <a id="check-data"></a>

# There's 81 columns in data

# In[ ]:


# Columns
print(len(data.columns))
data.columns


# Check what types of data in each columns

# In[ ]:


# Data example
data.sample(n=10)


# Check types of each variables

# In[ ]:


data.dtypes.sort_values(ascending=False)


# For data pre-processing, categorize variables into 'Numerical Variables', 'Categorical Variables(int)', 'Categorical Variables(string)'  
#   
# **Numerical Variables: **float and int variables  
# ['MSSubClass', 'LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'YrSold']  
#   
# **Categorical Variables(int): **some of int variables  
# ['OverallQual', 'OverallCond', 'MoSold']  
#   
# **Categorical Variables(string): **string variables  
# ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

# Check how many data is missing

# In[ ]:


# Check missing values
def check_missing(df):
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum()/len(df)
    missing_table = pd.concat([null_val, percent], axis=1)
    col = missing_table.rename(columns = {0 : '#', 1 : '%'})
    return col

# Display columns missing values are under 1%.
print("Data #"+str(len(data)))
cols = check_missing(data)
# available_cols = cols[cols['%'] < 1]
# print(available_cols)
print(cols.sort_values(by="%", ascending=False))


# # Data Pre-Processing <a id="pre-processing"></a>

# Drop variables more than 40% data was missing..

# In[ ]:


# Drop more than 40% missing variables
data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1, inplace = True)


# Process categorical variables(string)
# 1. Fill missing data by most frequent value
# 2. Replace with dummy data

# In[ ]:


# Fill missing data and replace with dummy value
categorical_variables_string =     ['MSZoning', 'Street', 'LotShape', 'LandContour', 
     'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 
     'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 
     'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 
     'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 
     'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
     'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 
     'Electrical', 'KitchenQual', 'Functional', 'GarageType', 
     'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 
     'SaleType', 'SaleCondition']

for v in categorical_variables_string:
    # Fill NaN with mode
    data[v] = data[v].fillna(data[v].mode()[0])
    # Categorize
    data[v] = pd.factorize(data[v])[0]


# Process categorical variables(int)
# 1. Do nothing, because there's no missing data

# In[ ]:


# There's no missing data
categorical_variables_int =     ['OverallQual', 'OverallCond', 'MoSold']


# Process numerical variables
# 1. Just fill missing data with average
# 2. Standardize values

# In[ ]:


# Fill missing data
numerical_variavles =     ['MSSubClass', 'LotFrontage', 'LotArea', 'YearBuilt', 
     'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 
     'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 
     'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 
     'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 
     'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 
     'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 
     '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'YrSold']

ss = StandardScaler()
for v in numerical_variavles:
    # Fill NaN with mode
    data[v] = data[v].fillna(data[v].mean())
    # Standardize values
    data[v] = ss.fit_transform(data[[v]])


# Data after processing is like this

# In[ ]:


# Data example
data.sample(n=10)


# In[ ]:


# Set data
train = data[:1460]
test  = data[1460:]


# # Training and Prediction <a id="training-prediction"></a>

# To select parameters for training, use feature selection library

# In[ ]:


possible_features = categorical_variables_string + categorical_variables_int + numerical_variavles

# Check feature importances
selector = SelectKBest(f_regression, len(possible_features))
selector.fit(train[possible_features], train['SalePrice'])
scores = -np.log10(selector.pvalues_)
indices = np.argsort(scores)[::-1]

print('Feature importances:')
for i in range(len(scores)):
    print('%.2f %s' % (scores[indices[i]], possible_features[indices[i]]))


# This time, pick variables by their importances  
#   
# **Possible features:**  
# ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF',  
# '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd',  
# 'MasVnrArea', 'GarageYrBlt', 'Fireplaces', 'Foundation', 'HeatingQC',  
# 'BsmtFinSF1', 'BsmtFinType1', 'LotFrontage', 'WoodDeckSF', '2ndFlrSF',  
# 'OpenPorchSF', 'HalfBath', 'LotShape', 'ExterQual', 'LotArea',  
# 'CentralAir', 'Electrical', 'BsmtExposure', 'BsmtFullBath', 'BsmtUnfSF',  
# 'PavedDrive', 'HouseStyle', 'BedroomAbvGr', 'Exterior2nd', 'RoofStyle',  
# 'Neighborhood', 'SaleCondition', 'GarageFinish', 'KitchenAbvGr', 'EnclosedPorch',  
# 'ExterCond', 'Exterior1st', 'MSZoning', 'KitchenQual', 'BldgType',  
# 'GarageCond', 'ScreenPorch', 'LotConfig', 'Functional', 'Heating',  
# 'GarageType', 'PoolArea', 'LandContour', 'MSSubClass', 'BsmtCond',  
# 'OverallCond', 'SaleType', 'GarageQual', 'LandSlope', 'BsmtFinType2',  
# 'MoSold', 'Condition1', '3SsnPorch', 'Street', 'RoofMatl',  
# 'YrSold', 'LowQualFinSF', 'MiscVal', 'BsmtHalfBath', 'Utilities',  
# 'BsmtFinSF2', 'MasVnrType', 'Condition2', 'BsmtQual']

# In[ ]:


# Feature params
fparams =     ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF',
    '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd',
    'MasVnrArea', 'GarageYrBlt', 'Fireplaces', 'Foundation', 'HeatingQC',
    'BsmtFinSF1', 'BsmtFinType1', 'LotFrontage', 'WoodDeckSF', '2ndFlrSF',
    'OpenPorchSF', 'HalfBath', 'LotShape', 'ExterQual', 'LotArea',
    'CentralAir', 'Electrical', 'BsmtExposure', 'BsmtFullBath', 'BsmtUnfSF',
    'PavedDrive', 'HouseStyle', 'BedroomAbvGr', 'Exterior2nd', 'RoofStyle',
    'Neighborhood', 'SaleCondition', 'GarageFinish', 'KitchenAbvGr', 'EnclosedPorch',
    'ExterCond', 'Exterior1st', 'MSZoning', 'KitchenQual', 'BldgType',
    'GarageCond', 'ScreenPorch', 'LotConfig', 'Functional', 'Heating',
    'GarageType', 'PoolArea', 'LandContour', 'MSSubClass', 'BsmtCond',
    'OverallCond', 'SaleType', 'GarageQual', 'LandSlope', 'BsmtFinType2',
    'MoSold', 'Condition1', '3SsnPorch']

# Get params
train_target = train["SalePrice"].values
train_features = train[fparams].values
test_features  = test[fparams].values


# Here's just use RandomForestRegressor for prediction, and do GridSearch

# In[ ]:


from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor

rfgs_parameters = {
    'n_estimators': [50],
    'max_depth'   : [n for n in range(2, 16)],
    'max_features': [n for n in range(2, 16)],
    "min_samples_split": [n for n in range(2, 8)],
    "min_samples_leaf": [n for n in range(2, 8)],
    "bootstrap": [True,False]
}

rfr_cv = GridSearchCV(RandomForestRegressor(), rfgs_parameters, cv=8, scoring= 'neg_mean_squared_log_error')
rfr_cv.fit(train_features, train_target)
print("RFR GridSearch score: "+str(rfr_cv.best_score_))
print("RFR GridSearch params: ")
print(rfr_cv.best_params_)


# Output prediction result to a file

# In[ ]:


prediction = rfr_cv.best_estimator_.predict(test_features)
pred = pd.DataFrame(pd.read_csv("../input/test.csv")['Id'])
pred['SalePrice'] = prediction
pred.to_csv("../working/submission.csv", index = False)

