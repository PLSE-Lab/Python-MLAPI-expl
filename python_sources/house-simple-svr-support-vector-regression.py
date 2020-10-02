#!/usr/bin/env python
# coding: utf-8

# ref: this notebook has been fork from https://www.kaggle.com/himaoka/house-simple-svr-support-vector-regression

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
data.sample(n=20)


# Check types of each variables

# In[ ]:


types = pd.DataFrame(data.dtypes).rename(columns={0: 'type'}).sort_values(by=['type'],ascending=False)
types


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
    col = missing_table.rename(columns = {0 : 'Num', 1 : 'Rate'})
    return col

# Display columns missing values are under 1%.
print("Data #"+str(len(data)))
cols = check_missing(data)
types.join(cols).sort_values(by="Rate", ascending=False)


# # Data Pre-Processing <a id="pre-processing"></a>

# Drop variables more than 40% data was missing..

# In[ ]:


# Drop more than 40% missing variables
data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1, inplace = True)


# Process categorical variables(string)
# 1. Fill missing data by most frequent value
# 2. One-Hot Encoding

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
    # One-Hot Encoding
    data = pd.get_dummies(data, columns=[v], drop_first=True)
    # Categorize
    # data[v] = pd.factorize(data[v])[0]


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
    # Fill NaN with mean
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


possible_features = train.columns.copy().drop('SalePrice').drop('Id')

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
# **Possible features(Ordered by importances)**  
#     ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', 
#      '1stFlrSF', 'ExterQual_TA', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 
#      'KitchenQual_TA', 'GarageFinish_Unf', 'YearRemodAdd', 'BsmtQual_TA', 
#      'Foundation_PConc', 'MasVnrArea', 'GarageYrBlt', 'Fireplaces', 
#      'ExterQual_Gd', 'BsmtFinType1_GLQ', 'Neighborhood_NridgHt', 'BsmtFinSF1', 
#      'MasVnrType_None', 'SaleType_New', 'GarageType_Detchd', 'SaleCondition_Partial', 
#      'Foundation_CBlock', 'LotFrontage', 'MasVnrType_Stone', 'Neighborhood_NoRidge', 
#      'WoodDeckSF', 'KitchenQual_Gd', 'BsmtExposure_No', '2ndFlrSF', 'OpenPorchSF', 
#      'HeatingQC_TA', 'BsmtExposure_Gd', 'Exterior2nd_VinylSd', 'Exterior1st_VinylSd', 
#      'MSZoning_RM', 'HalfBath', 'LotShape_Reg', 'LotArea', 'CentralAir_Y', 'MSZoning_RL', 
#      'HouseStyle_2Story', 'SaleType_WD', 'Electrical_SBrkr', 'RoofStyle_Hip', 'GarageType_BuiltIn', 
#      'BsmtQual_Gd', 'GarageType_Attchd', 'PavedDrive_Y', 'BsmtFullBath', 'RoofStyle_Gable', 
#      'Neighborhood_StoneBr', 'BsmtUnfSF', 'MasVnrType_BrkFace', 'Neighborhood_OldTown', 
#      'Neighborhood_NAmes', 'Neighborhood_Edwards', 'GarageFinish_RFn', 'RoofMatl_WdShngl', 
#      'BedroomAbvGr', 'Exterior1st_MetalSd', 'Neighborhood_IDOTRR', 'Exterior2nd_MetalSd', 
#      'Exterior2nd_Wd Sdng', 'Exterior1st_Wd Sdng', 'KitchenQual_Fa', 'SaleCondition_Normal', 
#      'Neighborhood_BrkSide', 'LotConfig_CulDSac', 'Neighborhood_Somerst', 'ExterCond_Fa', 
#      'GarageCond_TA', 'KitchenAbvGr', 'BsmtFinType1_Rec', 'HeatingQC_Gd', 'HeatingQC_Fa', 
#      'Exterior1st_CemntBd', 'GarageQual_Fa', 'BsmtFinType1_Unf', 'BsmtFinType1_BLQ', 
#      'GarageCond_Fa', 'BsmtQual_Fa', 'EnclosedPorch', 'Neighborhood_Sawyer', 'Exterior2nd_CmentBd', 
#      'Electrical_FuseF', 'Neighborhood_Timber', 'LotShape_IR2', 'LandContour_HLS', 'Foundation_Slab', 
#      'Condition1_Feedr', 'Functional_Typ', 'ExterQual_Fa', 'BldgType_Duplex', 'Condition1_Norm', 
#      'Neighborhood_MeadowV', 'ScreenPorch', 'ExterCond_TA', 'RoofMatl_CompShg', 'Neighborhood_BrDale', 
#      'BldgType_Twnhs', 'BldgType_2fmCon', 'GarageQual_TA', 'Exterior1st_HdBoard', 'HouseStyle_SFoyer', 
#      'Heating_GasA', 'PoolArea', 'Heating_Grav', 'MSZoning_FV', 'BsmtCond_Gd', 'PavedDrive_P', 
#      'HouseStyle_1.5Unf', 'BsmtFinType1_LwQ', 'MSSubClass', 'LotConfig_Inside', 'OverallCond', 
#      'Exterior2nd_ImStucc', 'Neighborhood_CollgCr', 'Functional_Min2', 'Neighborhood_Crawfor', 
#      'GarageType_CarPort', 'Functional_Maj2', 'Exterior2nd_HdBoard', 'MSZoning_RH', 'Functional_Min1', 
#      'Neighborhood_SWISU', 'Neighborhood_Veenker', 'GarageCond_Po', 'HouseStyle_1Story', 'Heating_Wall', 
#      'Neighborhood_Mitchel', 'BsmtFinType2_BLQ', 'BsmtFinType2_Unf', 'Neighborhood_ClearCr', 
#      'BsmtCond_Po', 'Exterior2nd_Plywood', 'Exterior1st_WdShing', 'Exterior1st_BrkComm', 
#      'SaleCondition_AdjLand', 'ExterCond_Gd', 'Condition1_PosN', 'Condition2_PosN', 'Condition2_Feedr', 
#      'Electrical_FuseP', 'Condition2_PosA', 'Exterior2nd_Brk Cmn', 'Condition1_RRAe', 
#      'SaleCondition_Family', 'MoSold', 'GarageQual_Po', 'LandContour_Low', 'Exterior2nd_Other', 
#      'RoofMatl_WdShake', '3SsnPorch', 'BsmtExposure_Mn', 'GarageQual_Gd', 'LandSlope_Mod', 
#      'Exterior2nd_Stucco', 'Condition1_PosA', 'SaleType_ConLD', 'SaleType_Con', 'Street_Pave', 
#      'Exterior2nd_Wd Shng', 'BsmtFinType2_Rec', 'Condition2_RRNn', 'HouseStyle_SLvl', 
#      'Neighborhood_NPkVill', 'BsmtFinType2_LwQ', 'Electrical_Mix', 'LotShape_IR3', 'HouseStyle_2.5Fin', 
#      'Exterior1st_Stone', 'Neighborhood_Gilbert', 'RoofStyle_Gambrel', 'SaleType_Oth', 'ExterCond_Po', 
#      'Exterior1st_BrkFace', 'HeatingQC_Po', 'Condition2_Norm', 'Exterior1st_Stucco', 'GarageType_Basment', 
#      'YrSold', 'LandSlope_Sev', 'LandContour_Lvl', 'SaleType_ConLw', 'Exterior1st_ImStucc', 
#      'Exterior1st_AsphShn', 'HouseStyle_2.5Unf', 'Heating_OthW', 'LowQualFinSF', 'Exterior2nd_CBlock', 
#      'Exterior1st_CBlock', 'BsmtCond_TA', 'Exterior2nd_BrkFace', 'Exterior2nd_AsphShn', 
#      'Neighborhood_NWAmes', 'Condition1_RRNn', 'MiscVal', 'RoofStyle_Shed', 'Neighborhood_Blueste', 
#      'Heating_GasW', 'RoofMatl_Membran', 'SaleType_CWD', 'LotConfig_FR3', 'Exterior1st_Plywood', 
#      'Functional_Sev', 'BsmtHalfBath', 'Exterior2nd_Stone', 'Functional_Mod', 'SaleCondition_Alloca', 
#      'Neighborhood_SawyerW', 'Condition2_RRAn', 'RoofMatl_Roll', 'SaleType_ConLI', 'Utilities_NoSeWa', 
#      'Foundation_Stone', 'BsmtFinSF2', 'LotConfig_FR2', 'Condition1_RRAn', 'RoofMatl_Tar&Grv', 
#      'Condition1_RRNe', 'BldgType_TwnhsE', 'Condition2_RRAe', 'Foundation_Wood', 'GarageCond_Gd', 
#      'RoofStyle_Mansard', 'RoofMatl_Metal', 'BsmtFinType2_GLQ']

# In[ ]:


# Feature params
fparams =     ['OverallQual', 'YearBuilt']

# Get params
train_target = train["SalePrice"].values
train_features = train[fparams].values
test_features  = test[fparams].values


# Here's just use SVR for prediction, with GridSearch

# In[ ]:


from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split

svrgs_parameters = {
    'kernel': ['rbf'],
    'C':     [150000,200000,250000],
    'gamma': [0.004,0.0045,0.005]
}

svr_cv = GridSearchCV(svm.SVR(), svrgs_parameters, cv=8, scoring= 'neg_mean_squared_log_error')
svr_cv.fit(train_features, train_target)
print("SVR GridSearch score: "+str(svr_cv.best_score_))
print("SVR GridSearch params: ")
print(svr_cv.best_params_)


# Output prediction result to a file

# In[ ]:


prediction = svr_cv.best_estimator_.predict(test_features)
pred = pd.DataFrame(pd.read_csv("../input/test.csv")['Id'])
pred['SalePrice'] = prediction
pred.to_csv("../working/submission.csv", index = False)


# In[ ]:





# In[ ]:





# In[ ]:




