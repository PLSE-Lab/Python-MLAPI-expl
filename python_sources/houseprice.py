#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
from sklearn.exceptions import DataConversionWarning
# Suppress warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)


# In[ ]:




import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler

# Set pandas data display option
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 500)


# Training data

# In[ ]:


train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv", index_col=0)
train.head()


# test data

# In[ ]:


test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv", index_col=0)
test.head()


# In[ ]:


train.shape , test.shape


# In[ ]:


train.info()


# In[ ]:


test.info()


# The training data has almost the same size as the test data!

# In[ ]:


train.shape[0] / test.shape[0]


# In[ ]:


train_test_data  = train.append(test, sort=False)


# In[ ]:


train_test_data.head()


# In[ ]:


train_test_data.info()


# In[ ]:


import pandas_profiling as pp

#Explore the data using pandas_profiling
profile = pp.ProfileReport(train_test_data)
profile


# In[ ]:


train_test_data.describe() 


# Submission

# In[ ]:


submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv", index_col=0)
submission.head()


# In[ ]:


submission.info()


# **
# Data exploration and Visulazition
# **

# **
# Target distribution
# **

# In[ ]:


plt.figure(figsize=(20,5))
sns.distplot(train.SalePrice, color="blue")
plt.title("Target distribution in train")
plt.ylabel("Probability Density Function");


# 
# Check negtaive value of saleprice 

# In[ ]:


train[train.SalePrice <=0 ]


# There is no negtaive values or equals to zero in saleprice

# 
# **Check Train_test_Data **

# In[ ]:


print(len(train_test_data.columns))
train_test_data.columns


# 
# 
# Check what types of data in each columns

# In[ ]:


train_test_data.sample(n=10)


# 
# 
# Relationship with numerical variables
# 
# Relationship of grlivarea wtih saleprice
# 
# Describe of GrLivArea: Above grade (ground) living area square feet
# 

# In[ ]:


#scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
sns.regplot(x=var, y='SalePrice',data=data)


# 
# 
# It seems that 'SalePrice' and 'GrLivArea' are really have a linear relationship.
# 

# In[ ]:


data.head()


# 
# 
# Relationship of TotalBsmtSF wtih saleprice
# 
# Describe of TotalBsmtSF: Total square feet of basement area
# 

# In[ ]:


#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
sns.regplot(x=var, y='SalePrice',data=data)


# 
# 
# It seems that 'SalePrice' and 'totalbsmtsf' are strong linear (exponential?) reaction, everything changes. Moreover, it's clear that sometimes 'TotalBsmtSF' closes in itself and gives zero credit to 'SalePrice'.
# 

# 
# 
# **Relationship with categorical features
# **
# 
# Relationship of OverallQual wtih saleprice
# 
# Describe of OverallQual: Rates the overall material and finish of the house 10 Very Excellent 9 Excellent 8 Very Good 7 Good 6 Above Average 5 Average 4 Below Average 3 Fair 2 Poor 1 Very Poor
# 

# In[ ]:


#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# 
# 
# We can see there is good increase relation between saleprice and overallquall
# 
# 

# Relationship of YearBuilt wtih saleprice
# 

# In[ ]:


var = 'YearBuilt'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);


# Although it's not a strong tendency, I'd say that 'SalePrice' is more prone to spend more money in new stuff than in old relics.
# 
# Note: we don't know if 'SalePrice' is in constant prices. Constant prices try to remove the effect of inflation. If 'SalePrice' is not in constant prices, it should be, so than prices are comparable over the years

# In[ ]:


train.corr()


# 
# 
# Correlation matrix (heatmap style)
# 

# In[ ]:


#correlation matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# 
# 
# 'SalePrice' correlation matrix (zoomed heatmap style)
# 

# In[ ]:


#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                 annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# 
# 
# Scatter plots between 'SalePrice' and correlated variables (move like Jagger style)
# 

# In[ ]:


#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size = 2.5)
plt.show();


# 
# 
# Check types of each variables
# 

# In[ ]:


train_test_data.dtypes.sort_values(ascending=False)


# For data pre-processing:
# categorize variables into :
# * Numerical Variables
# * Categorical Variables
#   *  Categorical Variables(int)
#   * Categorical Variables(string)

# **Numerical Variables**: float and int variables
# ['MSSubClass', 'LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'YrSold']

# **Categorical Variables(int)**: some of int variables
# ['OverallQual', 'OverallCond', 'MoSold']
# 
# **Categorical Variables(string)**: string variables
# ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

# **Missing Data**

# In[ ]:


#missing data
total = train_test_data.isnull().sum().sort_values(ascending=False)
percent = (train_test_data.isnull().sum()/train_test_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(80)


# In[ ]:


missing = train_test_data.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()


# **
# Data Pre-Processing**

# 
# 
# Drop variables more than 40% data was missing..
# 

# In[ ]:


# Drop more than 40% missing variables
train_test_data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1, inplace = True)


# 
# 
# Process categorical variables(string)
# 
#     1.Fill missing data by most frequent value
#     2.Replace with dummy data
# 
# 

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
    train_test_data[v] = train_test_data[v].fillna(train_test_data[v].mode()[0])
    # Categorize
    train_test_data[v] = pd.factorize(train_test_data[v])[0]


# 
# 
# Process categorical variables(int)
# 
#     Do nothing, because there's no missing data
# 
# 

# In[ ]:


# There's no missing data
categorical_variables_int =     ['OverallQual', 'OverallCond', 'MoSold']


# 
# 
# Process numerical variables
# 
#     1.Just fill missing data with average
#     2.Standardize values
#     
# 
# 

# In[ ]:


# Fill missing data
numerical_variables =     ['MSSubClass', 'LotFrontage', 'LotArea', 'YearBuilt', 
     'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 
     'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 
     'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 
     'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 
     'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 
     'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 
     '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'YrSold']

ss = StandardScaler()
for v in numerical_variables:
    # Fill NaN with mode
    train_test_data[v] = train_test_data[v].fillna(train_test_data[v].mean())
    # Standardize values
    train_test_data[v] = ss.fit_transform(train_test_data[[v]])


# Data after processing is like this

# In[ ]:


# Data proceesing
train_test_data.sample(n=10)


# In[ ]:


# Set data
train = train_test_data[:1460]
test  = train_test_data[1460:]


# **Training and Prediction**

# 
# 
# To select parameters for training, use feature selection library
# 

# In[ ]:


possible_features = categorical_variables_string + categorical_variables_int + numerical_variables

# Check feature importances
selector = SelectKBest(f_classif, len(possible_features))
selector.fit(train[possible_features], train['SalePrice'])
scores = -np.log10(selector.pvalues_)
indices = np.argsort(scores)[::-1]

print('Feature importances:')
for i in range(len(scores)):
    print('%.2f %s' % (scores[indices[i]], possible_features[indices[i]]))


# 
# 
#  Use variables that its importance was more than 1.
# 

# In[ ]:


# Feature params
fparams =     ['OverallQual', 'MiscVal', 'GrLivArea', 'LotArea', 'GarageCars', 
     'FullBath', 'ExterQual', 'GarageArea', '1stFlrSF', 'TotalBsmtSF', 
     'YearBuilt', 'Foundation', 'KitchenQual', 'MasVnrArea', 'BsmtQual', 
     'TotRmsAbvGrd', 'Street', 'YearRemodAdd', 'GarageYrBlt', 'BsmtFinSF1', 
     '2ndFlrSF', 'Fireplaces', 'Heating', 'CentralAir', 'LotShape', 
     'SaleCondition', 'BsmtUnfSF', 'OpenPorchSF', 'GarageFinish', 'HalfBath', 
     'WoodDeckSF', 'HeatingQC', 'BsmtExposure', 'LandContour', 'Exterior2nd', 
     'MSZoning', 'BedroomAbvGr', 'MasVnrType', 'BsmtFinType1', 'LotFrontage', 
     'Neighborhood', 'Electrical', 'LandSlope', 'SaleType', 'BsmtFullBath', 
     'Exterior1st', 'OverallCond']

# Get params
train_target = train["SalePrice"].values
train_features = train[fparams].values
test_features  = test[fparams].values


# Model

# In[ ]:


from xgboost import XGBRegressor
xgboost = XGBRegressor(max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)


# In[ ]:


from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split

svrgs_parameters = {
    'kernel': ['rbf'],
    'C':     [200000,210000,220000,230000,240000,250000],
    'gamma': [0.003,0.00325,0.0035,0.00375,0.004]
}

svr_cv = GridSearchCV(svm.SVR(), svrgs_parameters, cv=8, scoring= 'neg_mean_squared_log_error')
svr_cv.fit(train_features, train_target)
print("SVR GridSearch score: "+str(svr_cv.best_score_))
print("SVR GridSearch params: ")
print(svr_cv.best_params_)


# Output submission

# In[ ]:


prediction = svr_cv.best_estimator_.predict(test_features)
output = pd.DataFrame(pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")['Id'])
output['SalePrice'] = prediction
output.to_csv("../working/submission.csv", index = False)


# In[ ]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_true=train_target[:1459], y_pred=prediction[:1459])


# In[ ]:


submission.head()

