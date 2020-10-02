#!/usr/bin/env python
# coding: utf-8

# # 1) Read in the Modules

# In[51]:


# Import the modules

import pandas as pd
import numpy as np
#from scipy import stats
import sklearn as sk
#import itertools

# Data Vis
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='white', context='notebook', palette='deep') 
import matplotlib.style as style
style.use('fivethirtyeight')


from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV

from scipy.stats import skew

# Data Scaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Regression
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet

# Metrics
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


# # 2) Load Data and Initial Inspection

# ## 2.1) Load Data and Quality Checks

# **We should try to keep training and test data separate so as not to influence our decisions by the test data, which should be used to test the generalisation of the model and therefore needs to remain 'unseen'. This also applies to the imputation of Null values.**

# In[52]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[53]:


# Change the settings so that you can see all columns of the dataframe when calling df.head()
pd.set_option('display.max_columns',999)
train.head()


# In[54]:


# Get the shape of the data

print(train.shape)
print(test.shape)


# ## 2.2) Initial Data Inspection

# **Let's get an initial idea of what variables we are dealing with, whether they contain missing data, and what their distribution looks like.**

# In[55]:


train.columns


# In[56]:


train.info()


# **Let's visualise the missing data for both train and test set**

# In[57]:


# Capture the necessary data
variables = train.columns

count = []

for variable in variables:
    length = train[variable].count()
    count.append(length)
    
count_pct = np.round(100 * pd.Series(count) / len(train), 2)
count = pd.Series(count)

missing = pd.DataFrame()
missing['variables'] = variables
missing['count'] = len(train) - count
missing['count_pct'] = 100 - count_pct
missing = missing[missing['count_pct'] > 0]
missing.sort_values(by=['count_pct'], inplace=True)
missing_train = np.array(missing['variables'])

#Plot number of available data per variable
plt.subplots(figsize=(15,6))

# Plots missing data in percentage
plt.subplot(1,2,1)
plt.barh(missing['variables'], missing['count_pct'])
plt.title('Count of missing training data in percent', fontsize=15)

# Plots total row number of missing data
plt.subplot(1,2,2)
plt.barh(missing['variables'], missing['count'])
plt.title('Count of missing training data as total records', fontsize=15)

plt.show()


# In[58]:


# Capture the necessary data
variables = test.columns

count = []

for variable in variables:
    length = test[variable].count()
    count.append(length)
    
count_pct = np.round(100 * pd.Series(count) / len(test), 2)
count = pd.Series(count)

missing = pd.DataFrame()
missing['variables'] = variables
missing['count'] = len(test) - count
missing['count_pct'] = 100 - count_pct
missing = missing[missing['count_pct'] > 0]
missing.sort_values(by=['count_pct'], inplace=True)
missing_test = np.array(missing['variables'])

#Plot number of available data per variable
plt.subplots(figsize=(15,6))

# Plots missing data in percentage
plt.subplot(1,2,1)
plt.barh(missing['variables'], missing['count_pct'])
plt.title('Count of missing test data in percent', fontsize=15)

# Plots total row number of missing data
plt.subplot(1,2,2)
plt.barh(missing['variables'], missing['count'])
plt.title('Count of missing test data as total records', fontsize=15)

plt.show()


# **Attributes like PoolQC and MiscFeatures have a lot of missing data in both training and test set and it might be neccessary to drop them.
# <br>
# On the other hand, a lot of the NULL values might not be missing data but it was simply not applicable. For example, if there is no pool then there cannot be any pool quality.
# <br>
# This is indeed the case when looking at the data dictionary and I will convert the values in the next section.**

# In[59]:


train.describe()


# In[60]:


test.describe()


# ## 2.3) Data Cleaning - dealing with missing data and incorrect data types

# ### 2.3.1) Change Data Types

# **To Objects**

# In[61]:


# Convert training data
train['MSSubClass'] = train['MSSubClass'].astype('object')
train['MoSold'] = train['MoSold'].astype('object')

# Convert test data
test['MSSubClass'] = test['MSSubClass'].astype('object')
test['MoSold'] = test['MoSold'].astype('object')


# **To numeric** - for simplicity I convert everything to float

# In[62]:


# Convert training data
numeric_train = train.select_dtypes(include=[np.number]).columns.tolist()
train[numeric_train] = train[numeric_train].astype('float64')

# Convert test data
numeric_test = test.select_dtypes(include=[np.number]).columns.tolist()
test[numeric_test] = test[numeric_test].astype('float64')


# ### 2.3.2) Get a list of variables that will be adjusted

# **In this step I get all variables that I will adjust and save them so that they are more easily accessible**

# **Get list of variables that are objects or numerics and whose NULL values will be overridden**

# In[63]:


#Exclude features with no NULL values

drop_object = ['Functional', 'SaleType', 'Exterior1st', 'Exterior2nd']
drop_numeric = ['GarageYrBlt']


train_object = train.columns[(train.dtypes == 'object') & (train.columns.isin(missing_train))]
test_object = test.columns[(test.dtypes == 'object') & (test.columns.isin(missing_test))].drop(drop_object)

train_numeric = train.columns[(train.dtypes != 'object') & (train.columns.isin(missing_train))].drop(drop_numeric) # if the garage was never build then we can't impute a date
test_numeric = test.columns[(test.dtypes != 'object') & (test.columns.isin(missing_test))].drop(drop_numeric)


# **Get list of variables that can be imputed**

# In[64]:


#train_object_impute = train[['Electrical', 'Utilities', 'MSZoning', 'KitchenQual', 'Exterior1st']].columns
test_object_impute = test[['Electrical', 'Utilities', 'MSZoning', 'KitchenQual', 'Exterior1st']].columns


# ### 2.3.5) Impute Null values

# In[65]:


# These are variables where it makes sense to impute them rather than setting them to 0 or 'Other'
# As they are categorical values I need to use the mode instead of the mean or median

# Impute training data
#for variable in train_object_impute:
#    train[variable] = train[variable].fillna(train[variable].mode()[0])

# Impute test data
for variable in test_object_impute:
    test[variable] = test[variable].fillna(test[variable].mode()[0])


# ### 2.3.3) Non-Numerical Data: Overwriting Nulls with 'Missing'

# Now I will overwrite all non-numerical data with whatever the appropriate 'Missing' value is, which can be 'None', 'Other' etc.

# In[66]:


# Get the non-numerical variables apart from 'Functional', 'SaleType', 'Exterior1st', 'Exterior2nd' as they get
# different values according to the data dictionary

for variable in train_object:
    train[variable] = train[variable].fillna('None')
    
train['Functional'] = train['Functional'].fillna('Typ') 
train['SaleType'] = train['SaleType'].fillna('Oth')
#train['Exterior1st'] = train['Exterior1st'].fillna('Other')
train['Exterior2nd'] = train['Exterior2nd'].fillna('Other')

# Do the same thing for test now

for variable in test_object:
    test[variable] = test[variable].fillna('None')
    
test['Functional'] = test['Functional'].fillna('Typ') 
test['SaleType'] = test['SaleType'].fillna('Oth')
#test['Exterior1st'] = test['Exterior1st'].fillna('Other')
test['Exterior2nd'] = test['Exterior2nd'].fillna('Other')


# ### 2.3.4) Numerical Data: Overwrite Nulls to 0

# In[67]:


# Get numerical values apart from GarageYrBlt, which can't be imputed if it was neer built. However if GarageCars or 
# GarageArea have positive values then I can impute GarageYrBlt 

for variable in train_numeric:
    train[variable] = train[variable].fillna(0)
    
for variable in test_numeric:
    test[variable] = test[variable].fillna(0)


# There are roughly 80 GarageYrBlt values missing in both train and test - I am not sure how to impute those.

# # 3) EDA 

# ## 3.1) Analyse Object Data

# In[68]:


# Get quantitative variables
quantitative = [f for f in train.columns if train.dtypes[f] != 'object']
quantitative.remove('SalePrice')
quantitative.remove('Id')

# Get qualitative data
qualitative = [f for f in train.columns if train.dtypes[f] == 'object']


# In[69]:


def countplot(x, **kwargs):
    sns.countplot(x=x)
    x=plt.xticks(rotation=90)
f = pd.melt(train, value_vars=qualitative)
g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False, size=5)
g = g.map(countplot, "value")


# In[70]:


def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=90)
f = pd.melt(train, id_vars=['SalePrice'], value_vars=qualitative)
g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False, size=5)
g = g.map(boxplot, "value", "SalePrice")


# ## 3.2) Analyse Numeric Data

# In[71]:


f = pd.melt(train, value_vars=quantitative)
g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False, size=5)
g = g.map(sns.distplot, "value")


# In[72]:


f = pd.melt(train, id_vars=['SalePrice'], value_vars=quantitative)
g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False, size=5)
#g = g.map(plt.scatter, "value", "SalePrice")
g = g.map(sns.regplot, "value", "SalePrice")


# ## 3.3) Visualise the Predicted Variable: SalePrice

# In[73]:


sns.distplot(train['SalePrice'])
plt.title('Histogram of SalePrice')
plt.show()


# ## 3.4) Correlation Matrix

# Let's see which variables have a high linear correlation with SalePrice.

# In[74]:


# Correlation Matrix

# Compute the correlation matrix
d= train
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, 
            square=True, linewidths=.5, annot=False, cmap=cmap)
plt.yticks(rotation=0)
plt.title('Correlation Matrix of all Numerical Variables')
plt.show()


# In[75]:


corr.loc['SalePrice'][0:].sort_values()


# # 4) Prepare the Data

# ## 4.1) Decide Which Features to Include in the Models

# In[76]:


# Get all features from the training set apart from Id and SalePrice
predictor_cols = train.columns.drop(['Id','SalePrice'])
predictor_cols


# ### 4.1.1) Delete Outliers

# In[77]:


#train = train[train['LotFrontage'] < 300]
#train = train[train['LotArea'] < 100000]
#train = train[train['BsmtFinSF1'] < 5000]
#train = train[train['TotalBsmtSF'] < 6000]
#train = train[train['1stFlrSF'] < 4000]
train = train[train['GrLivArea'] < 4500]


# ### 4.1.2) Overwrite Bad Data 

# In[78]:


test[test['GarageYrBlt'] == 2207]['GarageYrBlt'] = 2007


# ## 4.2) Change Categorical Features to Ordinal Features

# This affects are 'quality features' and is based on the data dictionary.

# In[79]:


# Get a dictionary that can be used to map the ordinal values to the quality values
dict_num = {'None': 1, 'NA': 1,'Po': 2, 'Fa': 3, 'TA': 4, 'Gd': 5, 'Ex': 6,
            'Unf': 2,'LwQ': 3, 'Rec': 4, 'BLQ': 5, 'ALQ': 6, 'GLQ': 7,
            'Unf': 2, 'RFn': 3, 'Fin': 4,
            'Sal': 2,'Sev': 3, 'Maj2': 4, 'Maj1': 5, 'Mod': 6, 'Min2': 7, 'Min1': 8, 'Typ': 9,
            'MnWw': 2, 'GdWo': 3, 'MnPrv': 4, 'GdPrv': 5,
            'Sev': 1, 'Mod': 2, 'Gtl': 3,
            'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}


# In[80]:


quality_variable = ['ExterQual', 
                  'ExterCond',
                  'BsmtQual',
                  'BsmtCond',
                  'BsmtExposure',
                  'BsmtFinType1',
                  'BsmtFinType2',
                  'HeatingQC',
                  'KitchenQual',
                  'Functional',
                  'FireplaceQu',
                  'GarageFinish',
                  'GarageQual',
                  'GarageCond',
                  'PoolQC',
                  'Fence',
                  'LandSlope']


# In[81]:


for variable in quality_variable:
    train[variable] = train[variable].map(dict_num).astype('int')
    test[variable] = test[variable].map(dict_num).astype('int')


# ## 4.3) Create New Features

# In[82]:


train['TotalSF'] = (train['BsmtFinSF1'] + train['BsmtFinSF2'] + train['1stFlrSF'] + train['2ndFlrSF'])
test['TotalSF'] = (test['BsmtFinSF1'] + test['BsmtFinSF2'] + test['1stFlrSF'] + test['2ndFlrSF'])

train['Total_Bathrooms'] = (train['FullBath'] + (0.5*train['HalfBath']) + train['BsmtFullBath'] + (0.5*train['BsmtHalfBath']))
test['Total_Bathrooms'] = (test['FullBath'] + (0.5*test['HalfBath']) + test['BsmtFullBath'] + (0.5*test['BsmtHalfBath']))

train['Total_PorchSF'] = (train['OpenPorchSF'] + train['3SsnPorch'] + train['EnclosedPorch'] + train['ScreenPorch'] + train['WoodDeckSF'])
test['Total_PorchSF'] = (test['OpenPorchSF'] + test['3SsnPorch'] + test['EnclosedPorch'] + test['ScreenPorch'] + test['WoodDeckSF'])

train['HasPool'] = train['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
test['HasPool'] = test['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

train['Has2ndFloor'] = train['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
test['Has2ndFloor'] = test['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

train['HasGarage'] = train['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
test['HasGarage'] = test['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

train['HasBsmt'] = train['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
test['HasBsmt'] = test['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

train['HasFireplace'] = train['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
test['HasFireplace'] = test['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)


# ## 4.4 Delete Variables that aren't useful

# In[83]:


delete_features = ['Street', 
                    'Alley',
                   'LandSlope',
                   'Utilities',
                   'Condition2',
                   'RoofMatl',
                   'Heating',
                   'LowQualFinSF',
                   'KitchenAbvGr',
                   'Functional',
                   '3SsnPorch',
                   'ScreenPorch',
                   'PoolArea',
                   'PoolQC',
                   'MiscFeature',
                   'MiscVal']

maybe_delete_features = ['LandContour',
                        'Condition1',
                        'OverallCond',
                        'BsmtFinType2',
                        'BsmtFinSF2',
                        'BsmtUnfSF',
                        'GarageQual',
                        'GarageCond',
                        'WoodDeckSF',
                        'OpenPorchSF',
                        'EnclosedPorch',
                        'Fence',
                        'MoSold',
                        'YrSold',
                        'SaleType',
                        'SaleCondition']


# In[84]:


#predictor_cols = predictor_cols.drop(delete_features)
#predictor_cols = predictor_cols.drop(maybe_delete_features)
predictor_cols = predictor_cols.drop('GarageYrBlt')
predictor_cols = predictor_cols.drop(['BsmtFinSF1', 'BsmtFinSF2', '1stFlrSF', '2ndFlrSF', 'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath',
'OpenPorchSF', '3SsnPorch', 'EnclosedPorch', 'ScreenPorch', 'WoodDeckSF', 'PoolArea', '2ndFlrSF', 'GarageArea',
'TotalBsmtSF', 'Fireplaces'])


# In[85]:


# Get the predicted variable
y = train['SalePrice']
y.head()


# ## 4.5) Scale and Standardise Numeric Data

# ### 4.5.1 Predicted Variable: SalePrice

# In[86]:


y = np.log1p(y)


# ### 4.5.2 Numeric Predictors

# I only take the log if the skew is above a certain threshold - here it is 0.5. This is based on the interpretations from here: https://brownmath.com/stat/shape.htm

# In[87]:


# Log of numeric variables and show which features are the most skewed
# I am not taking into account the 'int' variables as they are ordinal categorical variables

numeric_features = train.dtypes[train.dtypes == 'float64'].index

skewed_features = train[numeric_features].apply(lambda x: skew(x))
np.abs(skewed_features).sort_values(ascending=False)


# In[88]:


# Continue taking the log when skewness > 0.5

skewed_features = skewed_features[skewed_features > 0.5]
skewed_features = skewed_features.index

train[skewed_features] = np.log1p(train[skewed_features])
test[skewed_features.drop('SalePrice')] = np.log1p(test[skewed_features.drop('SalePrice')])


# ## 4.6) One-Hot Encoding of Categorical Variables

# In[89]:


# 1)
pre_X_train = train[predictor_cols]
pre_X_test = test[predictor_cols]

# 2)
pre_X_train_one_hot = pd.get_dummies(pre_X_train)
pre_X_test_one_hot = pd.get_dummies(pre_X_test)

# 3)
X_train, X_test = pre_X_train_one_hot.align(pre_X_test_one_hot, join='inner', axis='columns')


# In[90]:


overfit = []
for i in pre_X_train_one_hot.columns:
    counts = pre_X_train_one_hot[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(pre_X_train_one_hot) * 100 >99:
        overfit.append(i)


# In[91]:


overfit = list(overfit)
overfit


# ## 4.7) Predicted Variable 

# In[92]:


y_train = y

print('X_train:', X_train.shape)
print('X_test:', X_test.shape)
print('y_train:', y_train.shape)


# ## 4.8) Quality Check of Prepared Data

# In[93]:


X_train.head()


# In[94]:


y_train


# In[95]:


X_test.head()


# In[96]:


# Capture the necessary data
variables = X_test.columns

count = []

for variable in variables:
    length = X_test[variable].count()
    count.append(length)
    
count_pct = np.round(100 * pd.Series(count) / len(X_test), 2)
count = pd.Series(count)

missing = pd.DataFrame()
missing['variables'] = variables
missing['count'] = len(X_test) - count
missing['count_pct'] = 100 - count_pct
missing = missing[missing['count_pct'] > 0]
missing.sort_values(by=['count_pct'], inplace=True)
missing_train = np.array(missing['variables'])

#Plot number of available data per variable
plt.subplots(figsize=(15,6))

# Plots missing data in percentage
plt.subplot(1,2,1)
plt.barh(missing['variables'], missing['count_pct'])
plt.title('Count of missing training data in percent', fontsize=15)

# Plots total row number of missing data
plt.subplot(1,2,2)
plt.barh(missing['variables'], missing['count'])
plt.title('Count of missing training data as total records', fontsize=15)

plt.show()


# # 5) Run Models

# ## Random Forest Regression

# In[97]:


rf_reg = RandomForestRegressor(n_estimators=100,
                              random_state=1)
rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_train)
cv_scores_rf = cross_val_score(rf_reg, X_train, y_train, cv=5)


# In[98]:


print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores_rf)))
# Print the 5-fold cross-validation scores
print(cv_scores_rf)


# In[99]:


rmse = np.sqrt(mean_squared_error(y_train, y_pred))
print("Root Mean Squared Error: {}".format(rmse))


# Feature Importance

# In[100]:


importance = pd.DataFrame(list(zip(X_train.columns, np.transpose(rf_reg.feature_importances_)))             ).sort_values(1, ascending=False)
importance


# In[101]:


importances = rf_reg.feature_importances_

std = np.std([tree.feature_importances_ for tree in rf_reg.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],  
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]),X_train.columns[indices], rotation=90)
#plt.xlim([-1, X_train_s.shape[1]])
plt.xlim([-1, 10])
plt.show()


# ## Random Forest GridSearch

# In[102]:


RandomForestRegressor().get_params().keys()


# In[103]:


rf_grid = RandomForestRegressor(n_estimators=1000,
                                max_depth=26, #16
                                max_features=40, #30
                                min_samples_leaf=10, # using 5 doesn't seem to improve things
                                random_state=1)
rf_grid.fit(X_train, y_train)
y_pred_grid = rf_grid.predict(X_train)
cv_scores_rf_grid = cross_val_score(rf_grid, X_train, y_train, cv=5)


# In[104]:


print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores_rf_grid)))
# Print the 5-fold cross-validation scores
print(cv_scores_rf_grid)


# In[105]:


rmse = np.sqrt(mean_squared_error(y_train, y_pred_grid))
print("Root Mean Squared Error: {}".format(rmse))


# ## LassoLarsCV

# In[106]:


from sklearn.linear_model import LassoLarsCV

lasso = LassoLarsCV(normalize=True)
lasso.fit(X_train, y_train)

y_pred = lasso.predict(X_train) #here not X_test used because somehow it doesn't work due to different array lengths
cv_scores_lasso = cross_val_score(lasso, X_train, y_train, cv=5)


# In[107]:


print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores_lasso)))
# Print the 5-fold cross-validation scores
print(cv_scores_lasso)


# In[108]:


rmse = np.sqrt(mean_squared_error(y_train, y_pred))
print("Root Mean Squared Error: {}".format(rmse))


# ## LassoLarsCV Gridsearch

# In[109]:


from sklearn.linear_model import LassoLarsCV

lars_grid = LassoLarsCV(normalize=False, fit_intercept=False, max_iter=200, max_n_alphas=500)
lars_grid.fit(X_train, y_train)

y_pred_grid = lars_grid.predict(X_train) #here not X_test used because somehow it doesn't work due to different array lengths
cv_scores_lars_grid = cross_val_score(lars_grid, X_train, y_train, cv=5)


# In[110]:


print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores_lars_grid)))
# Print the 5-fold cross-validation scores
print(cv_scores_lars_grid)


# In[111]:


rmse = np.sqrt(mean_squared_error(y_train, y_pred_grid))
print("Root Mean Squared Error: {}".format(rmse))


# ## XGBoost

# In[112]:


import xgboost as xgb

xgb = xgb.XGBRegressor(
                 colsample_bytree=0.2,  # fraction of features used in tree
                 gamma=0.0,             # node is only split if there is a positive loss reduction
                 learning_rate=0.01,
                 max_depth=4,
                 min_child_weight=1.5,  # Defines the minimum sum of weights of all observations required in a child. Larger numbers more conservative but can lead to underfitting
                 n_estimators=7200,                                                                  
                 reg_alpha=0.9,         # L1 regularization term on weight (analogous to Lasso regression
                 reg_lambda=0.6,        # L2 regularization term on weights (analogous to Ridge regression
                 subsample=0.2,         #very small number, maybe should be increased to 0.5-1
                 seed=42,
                 silent=0)              # supposed to give out messages. previously 1

xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_train)
cv_scores_xgb = cross_val_score(xgb, X_train, y_train, cv=5)


# In[113]:


print("Average 5-fold CV Score: {}".format(np.mean(cv_scores_xgb)))
# Print the 5-fold cross-validation scores
print(cv_scores_xgb)


# In[114]:


rmse = np.sqrt(mean_squared_error(y_train, y_pred))
print("Root Mean Squared Error: {}".format(rmse))


# ## Lasso

# In[115]:


from sklearn.linear_model import Lasso

# I found this best alpha through cross-validation.
best_alpha = 0.00099

lasso_regr = Lasso(alpha=best_alpha, max_iter=50000)
lasso_regr.fit(X_train, y_train)

y_pred = lasso_regr.predict(X_train)
cv_scores_lasso_regr = cross_val_score(lasso_regr, X_train, y_train, cv=5)


# In[116]:


print("Average 5-fold CV Score: {}".format(np.mean(cv_scores_lasso_regr)))
# Print the 5-fold cross-validation scores
print(cv_scores_lasso_regr)


# In[117]:


rmse = np.sqrt(mean_squared_error(y_train, y_pred))
print("Root Mean Squared Error: {}".format(rmse))


# ## Lasso GridSearch

# In[118]:


from sklearn.linear_model import Lasso

# I found this best alpha through cross-validation.
best_alpha = 0.00099

lasso_grid = Lasso(alpha=0.0001, max_iter=100, random_state=1)
lasso_grid.fit(X_train, y_train)

y_pred_grid = lasso_grid.predict(X_train)
cv_scores_lasso_grid = cross_val_score(lasso_grid, X_train, y_train, cv=5)


# In[119]:


print("Average 5-fold CV Score: {}".format(np.mean(cv_scores_lasso_grid)))
# Print the 5-fold cross-validation scores
print(cv_scores_lasso_grid)


# In[ ]:


rmse = np.sqrt(mean_squared_error(y_train, y_pred_grid))
print("Root Mean Squared Error: {}".format(rmse))


# # 6. Make Prediction on Test Set and Create CSV to be Submitted

# In[ ]:


y_pred_test = (xgb.predict(X_test) + lasso_regr.predict(X_test) + lasso_grid.predict(X_test)) / 3 
y_pred_test = np.expm1(y_pred_test)


# In[ ]:


submission = pd.DataFrame({'Id': test.Id.astype('int'), 
                           'SalePrice': y_pred_test})
#submission.to_csv('submission_ames.csv', index=False)
submission.shape

