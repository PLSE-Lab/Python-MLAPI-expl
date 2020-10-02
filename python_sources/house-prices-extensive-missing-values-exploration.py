#!/usr/bin/env python
# coding: utf-8

# In this project we are going to try to predict the house prices in Ames, Iowa. The dataset consists of roughly 3000 houses, split equally in training and test set. Each house in the dataset is described by 80 variables that include any information that a house listing could possibly include. 
# 
# The structure of this notebook is the following:
# In the first section, the development environment is set up, the train and test sets are loaded and we are taking a quick look at the dataframes. <br/>
# 
# In the second section, the datasets get explored.We are looking for possible outliers that will skew the models during training, and are investigating what is the best way to fill in each missing value.<br/>
# 
# Section 3 is handling data cleaning. This includes removing outliers from the data, filling in missing values and categorifying ordinal variables.<br/>
# 
# Finally, sections 4 and 5 revolve around developing a model, training it on the data and evaluating the results. The idea of using multiple regressors and averaging over their results comes from the following kernel: 
# 
# [Stacked Regressions : Top 4% on LeaderBoard](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard).

# # 1. Setting up environment

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


color = sns.color_palette()
sns.set_style('darkgrid')


# In[ ]:


path = '/kaggle/input/house-prices-advanced-regression-techniques'
path = Path(path)
train_raw = pd.read_csv(path/'train.csv')
test_raw = pd.read_csv(path/'test.csv')

train = train_raw.copy(deep=True)
test = test_raw.copy(deep=True)
data_clean = [train_raw,test_raw]


# In[ ]:


print("Dataset dimensions: ")
print("Training Set: " ,train.shape)
print("Test Set: " ,test.shape)


# In[ ]:


train.head(n=10)


# In[ ]:


test.head(n=10)


# In[ ]:


print("Variables with missing values in the dataset:")
print("Training set: " ,train.isnull().any().sum())
print("Test set: ", test.isnull().any().sum())


# We shall now save the house prices from the training set and unify train and test set to explore and fill in missing values.

# In[ ]:


y_train = train['SalePrice']
x_train = train.drop('SalePrice',axis = 1)

data = pd.concat([x_train,test],ignore_index= True, verify_integrity = True,copy = True)
print(data.shape)


# # 2. Data Exploration

# ## 2.1 Outliers

# Outliers in regression models can skew the performance significantly, since many advanced models are very sensitive to them. Therefore, we are looking for houses that were sold at a price lower than the expected value of the dataset. To achieve that, we plot the SalePrice of each house in the training set over the Lot Area and the above-ground living area.

# In[ ]:


train.plot.scatter(x = 'LotArea',y = 'SalePrice')
train.plot.scatter(x = 'GrLivArea',y = 'SalePrice')


# In the first plot, it cannot be concluded whether the points on the right part should be considered outliers. A bigger Lot Area is usually found in agricultural areas where the prices are lower than in urban areas. However, the 2 points on the lower right part of the second figure are definitely outliers, sold at a lower price than they should have been, and will be removed from the dataset.

# In[ ]:


idx_outliers =train[['GrLivArea','SalePrice']][(train['GrLivArea']>4000) & (train['SalePrice']<300000)]


# ## 2.2 Correlation of variables to house price
# 

# By computing the correlation between the numerical features of the data to the sale price, we get a first rough idea of which numerical variables are the most relevant to the target.

# In[ ]:


pd.options.display.max_rows = 50
price_corr = train[train.notnull()].corr(method='pearson')['SalePrice'].abs()
price_corr = pd.DataFrame(price_corr)
price_corr.sort_values(by = 'SalePrice',ascending = False)


# ## 2.3 Missing values

# The most important data cleaning procedure is to understand where missing values lie in the data and decide how to fill them.

# In[ ]:


mv = data.isnull().sum()/data.shape[0]*100
mv = mv[mv>0]
mv = mv.sort_values(axis = 0,ascending = False)


# In[ ]:


f, ax = plt.subplots(figsize=(15, 10))
plt.xticks(rotation='90')
sns.barplot(mv.index,mv)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
plt.show()


# Some missing values may be dependent on each other. For example, GarageQual is logically missing if there is no garage and pool quality is also logically missing when the proberty does not have a pool. This explains why some features have such a high percentage of missing values. Therefore, we have to go through each missing value variable one at a time to create a strategy on filling the missing values in a meaningful way.

# ### 2.3.1 Garage-related
# 
# There are 7 variables in the dataset with missing values related to the garage. We shall examine them all together. According to the documentation, if GarageType is NaN,then the property does not contain any garage space. 

# In[ ]:


garage = ['GarageArea', 'GarageCars', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'GarageYrBlt']


# In[ ]:


idx = data['GarageType'].isnull()
print("Total properties without a garage: ",idx.sum())
print("Non-NaN entries in properties without a garage: ")
print(data[garage][idx].notnull().sum())


# 5 of the 7 variables have no values entered in the aforementioned categories, while GarageArea and GarageCars are entried for all those properties.

# In[ ]:


idx = data['GarageType'].isnull() & data['GarageArea'].notnull() & data['GarageCars'].notnull()
print("Total Garage Area and Cards in properties without a garage:")
print(data[['GarageCars','GarageArea']][idx].sum())


# We see that those values were set to 0 even though no garage was mentioned, which makes sense. Therefore, no abnormality in the data is detected. 

# In[ ]:


idx = data['GarageArea'].isnull()
data[['GarageArea','GarageCars','GarageQual']][idx]


# There is one property where GarageArea and GarageCars are not entered, and since there is no Garage, they shall be set to 0.

# ### 2.3.2 Basement-related

# The second major category of missing values are related to basements. We will follow the same procedure as above to examine them. We will also later create a variable called hasBsmt to denote whether the building has a basement or not. This information can be extracted from the BsmtQual variable.

# In[ ]:


bsmt = ['BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual', 'BsmtUnfSF','TotalBsmtSF']
idx = data['BsmtQual'].isnull()
data[bsmt][idx].notnull().sum()


# This is weird and probably means that there may be something wrong with the data. Let's examine this further. First we consider the BsmtCond variable.

# In[ ]:


idx = data['BsmtQual'].isnull() & data['BsmtCond'].notnull()
data[bsmt][idx]


# Looking at the dataset description, these 2 entries have an unfinished basement without exposure, while the basement condition is fair in one case and typical in the other.Fair and typical are defined as:
# 
# * Typical - slight dampness allowed
# * Fair - dampness or some cracking or settling
# 
# Since these 2 properties do have a basement, even though it is small and unfinished, the hasBsmt variable should be set to 1. Additionally, it seems that these 2 basements are actually more fit to be used as storage space rather than as a functional part of the house. Therefore, they shall be filled with the median quality of unfinished basements with no exposure.
# 
# 

# Next we shall take a look at the variables BsmtFinSF1 and BsmtFinSF2.

# In[ ]:


idx = data['BsmtQual'].isnull() & data['BsmtFinSF1'].notnull()
data[bsmt][idx]


# In[ ]:


print(data[['BsmtFinSF1','BsmtFinSF2']][idx].sum())


# As we can see, the entries are just zeros. Nothing out of the ordinary here as well.

# In[ ]:


idx = data['BsmtFinSF2'].isnull()
data[['BsmtCond','BsmtFinSF1','BsmtFinSF2']][idx]


# Since there is no basement here either, they also should be set to 0.

# In[ ]:


idx = data['BsmtQual'].isnull() & data['BsmtFullBath'].notnull()
data[['BsmtFullBath','BsmtHalfBath']][idx].sum()


# The same applies for baths in the basement.

# In[ ]:


idx = data['BsmtQual'].isnull() & data['BsmtUnfSF'].notnull()
data[['BsmtUnfSF','TotalBsmtSF']][idx].sum()


# In[ ]:


idx = data['BsmtQual'].isnull() & data['BsmtUnfSF']>0
data[['BsmtUnfSF','TotalBsmtSF']][idx]


# These are the same 2 properties that we saw before. We can safely leave them as they are, as there is no abnormality here either.

# In[ ]:


pd.options.display.max_rows = 15
idx = data['BsmtExposure'].isnull() & data['BsmtCond'].notnull()
data[bsmt][idx]


# For these properties, BsmtExposure is set to indicate that no basement is present, even though there is a basement that is just unfinished. We are going to assume that the basement has no exposure. 

# In[ ]:


idx = data['BsmtCond'].notnull() & data['BsmtFinType1'].isnull()
data[bsmt][idx]


# In[ ]:


idx = data['BsmtCond'].notnull() & data['BsmtFinType2'].isnull()
data[bsmt][idx]


# In the one case where BsmtFinType2 is missing when a basement exists, we see that the basement has 1600 square feet of unfinished space. Therefore, BsmtFinType2 shall be set to unfinished.

# In[ ]:


idx = data['BsmtCond'].notnull() & data['BsmtFullBath'].isnull()
print(data[bsmt][idx])
idx = data['BsmtCond'].notnull() & data['BsmtHalfBath'].isnull()
print(data[bsmt][idx])


# Bath variables related to basements are only missing when there is no basement. Therefore, nothing special has to be done for them.

# In[ ]:


idx = data['BsmtCond'].notnull() & data['BsmtUnfSF'].isnull()
data[bsmt][idx]


# BsmtUnfSF is also only missing when there is no basement, therefore its missing values are set to 0.

# In[ ]:


idx = data['BsmtCond'].notnull() & data['TotalBsmtSF'].isnull()
data[bsmt][idx]


# Since no basement has a missing TotalBsmtSF value, it is only missing when no basement is present. Therefore, the missing values will be set to 0.

# ### 2.3.3 Other variables

# There are also some variables where, according to the documentation, a missing value denotes the absence of the corresponding feature. For example, if Fence is NaN for a property, then the property has no fence. These aren't really missing values and we will deal with them later when we categorify ordinal variables. 

# The documentation instructs to assume typical functionality unless deductions are warranted. Therefore, the missing values in Functional will be filled in with Typ. As for the Electrical and KitchenQual variables, they will be filled with the median of the corresponding house zoning (found in MSZoning). Finally, Utilities will be filled in with the median according to Functional. When LotFrontage is missing, it is assumed that no lot frontage is present and therefore is set to 0.

# Moving to the Exterior 1st and 2nd variables that describe the exterior of the house, we would like to check whether the Exterior2nd variable is obsolete. Therefore, we would like to see in how many properties Exterior2nd differs from Exterior1st. However, Exterior2nd values have in some cases a different name than in Exterior1st for the same thing. This should be fixed. These values are:
# 
# * Brk Cmn --> BrkComm
# * CmentBd --> CemntBd
# * Wd Shing  --> WdShing

# In[ ]:


dict_ext = {'Brk Cmn':'BrkComm',
            'CmentBd':'CemntBd',
            'Wd Shng': 'WdShing'
           }
data['Exterior2nd'] = data['Exterior2nd'].replace(dict_ext)


# In[ ]:


v = ['Exterior1st','Exterior2nd']
idx = data['Exterior1st']!=data['Exterior2nd']
data[v][idx]


# About ~9% of the properties have different exterior materials, which is enough to keep the variable. Moving on to see what is going on with the missing values.

# In[ ]:


pd.options.display.max_columns = 100

idx = data['Exterior1st'].isnull()
data[idx]


# In the one property with missing exterior values, they will be set according to the most frequent exterior cover for this neighborhood.

# In[ ]:


idx = data['MasVnrType'].isnull()
data[['MasVnrArea','MasVnrType']][idx]


# MasVnrArea's missing values will be set to 0, while MasVnrType will be set to None. In the one case where a veneer area is entered without a veneer type, the area will be corrected to 0.

# In[ ]:


data[data['MSZoning'].isnull()]


# The missing MSZoning values can be will with the median MSZoning value grouped by neighborhood, as properties in the same neighborhood should fall into the same zoning classification.

# In[ ]:


data[data['SaleType'].isnull()]


# Finally, the missing sale type value should be irrelevant to the final price of each property and will be dropped.

# # 3. Data Cleaning

# Now that all missing values have been explored, we are in a position to start cleaning the dataset.

# ## 3.1 Dropping outliers

# To avoid  any error, we will first drop the outliers from the training set and then re-initialize the 'data' dataframe to fill in the missing values and convert to ordinal data.

# In[ ]:


train = train.drop(idx_outliers.index, axis = 0)


# In[ ]:


train.plot.scatter(x = 'GrLivArea',y = 'SalePrice')


# Now the sale prices have a much clearer correlation and fitting a regression model on it will not be skewed by outliers. We can now recreate the 'data' dataframe and start cleaning the data.

# In[ ]:


y_train = train['SalePrice']
x_train = train.drop('SalePrice',axis = 1)

del data
data = pd.concat([x_train,test],ignore_index= True, verify_integrity = True,copy=True)


# In[ ]:


dict_ext = {'Brk Cmn':'BrkComm',
            'CmentBd':'CemntBd',
            'Wd Shng': 'WdShing'
           }
data['Exterior2nd'] = data['Exterior2nd'].replace(dict_ext)


# ## 3.2 Filling in missing values and fix wrong entries

# Before we convert the data to datatypes fit for training, the missing values should be filled in according to the analysis presented above. At the same time, any "wrong" data will be fixed. The whole process will be encapsulated into a function to ease readability and enable code reproducability.

# In[ ]:


def set_value(df,value, variables):
    assert type(variables)==list,"variables must be passed on as list"
    var0,var1 = variables
    idx = df[var0].notnull() & data[var1].isnull()
    loc = df[var1][idx].index[0]
    df.at[loc,var1] = value
    return df

def findStringMostCommon(d,target,conds, tvals=None):
    assert type(conds) == list, "Targetvars must be passed on as a list"
    assert len(conds)<3
    if tvals:
        if len(conds)>1:
            cond0,cond1 = conds
            tval0,tval1 = tvals    
            selected_data = d[target][(d[cond0]==tval0) & (d[cond1]==tval1)].sort_values()
        else:
            selected_data = d[target][d[conds]==tvals].sort_values()
    else:
        conds = conds[0]
        selected_data = d[target].groupby(conds).value_counts()
    
    return selected_data.value_counts().index[0]

def set_conditional(df,target,cond):
    idx = df[target].isnull()
    for i in idx.index:
        if idx[i]:
            stats = df[target][df[cond]==df.loc[i,cond]].value_counts()
            df.at[i,target] = stats.index[0]
    return df


# In[ ]:


def fill_and_fix(df):
    df_cp = df.copy(deep= True)
    vars_cat = ['Alley','BsmtCond','BsmtFinType1','BsmtFinType2','Fence','FireplaceQu','GarageCond','GarageFinish','GarageQual','GarageType','MiscFeature','PoolQC','BsmtQual']
    
    vars_num = ['BsmtFinSF1','BsmtFinSF2','BsmtFullBath','BsmtHalfBath','BsmtUnfSF','GarageArea','GarageCars','LotFrontage', 'TotalBsmtSF','MasVnrArea','GarageYrBlt']
    df_cp = set_value(df_cp,'Unf',['BsmtQual','BsmtFinType2'])
    df_cp = set_value(df_cp,findStringMostCommon(df_cp,'BsmtQual',['BsmtExposure','BsmtFinType1'],['No','Unf']),['BsmtCond','BsmtQual'])
    
    for v in ['Exterior1st','Exterior2nd','MSZoning','Utilities']:
        df_cp = set_conditional(df_cp,v,'Neighborhood')
        
    for v in ['Electrical','KitchenQual']:
        df_cp = set_conditional(df_cp,v,'MSZoning')
     
    idx = df_cp['MasVnrType'].isnull() & df_cp['MasVnrArea'].notnull()
    df_cp.at[idx[idx==True].index[0],'MasVnrArea']= 0 
    
    for var in vars_cat:
        df_cp[var].fillna(value = '0',inplace = True)
        
    for var in vars_num:
        df_cp[var].fillna(value = 0,inplace = True)
        
    df_cp['BsmtExposure'].fillna(value = '0',inplace = True)
    df_cp['Functional'].fillna(value = 'Typ',inplace = True) 
    df_cp['MasVnrType'].fillna(value = 'None',inplace = True)     
    
    return df_cp


# In[ ]:


data  = fill_and_fix(data)
print("Missing values after cleaning: ",data.isnull().sum().sum())


# The only missing value in the data is in the SaleType variable, which will later be dropped since it should make no difference on the house prices.

# 
# 

# ## 3.3 Feature Engineering

# Before we continue, we are going to crease some features and revamp some others. We are adding 3 variables: 
# 
# * TotalSF: The living area of a property, including 1st floor, 2nd floor and basement.
# * hasGarage: Variable indicating whether the property has a garage
# * hasBsmt: Variable indicating whether the property has a basement
# 
# We are also changing variables that indicate the age of some features of the house.
# * GarageAge: Replaces GarageYrBlt and equals to YearSold-GarageYrBlt
# * HouseAge: Replaces YearBlt, equals to YearSold-YearBuilt
# * RemodAge: Replaces YearRemodADd, equals to YearSold-YearRemodAdd
# 

# In[ ]:


data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']


# In[ ]:


data['hasGarage'] =  np.where(data['GarageQual']!='0', 1, 0)
data['hasBsmt'] =  np.where(data['BsmtCond']!='0', 1, 0)

data['GarageAge'] = data['YrSold']-data['GarageYrBlt']
data['GarageAge'] = np.where(data['GarageAge']<0,100,0)
data['HouseAge'] = data['YrSold']-data['YearBuilt']
data['RemodAge'] = data['YrSold']-data['YearRemodAdd']

data_clean = data.drop(labels = ['SaleType','GarageYrBlt','YearBuilt','YearRemodAdd'],axis = 1)


# ## 3.4 Converting categorical to ordinal data

# Most variables in the dataset are categorical. In order to train a model, these should be converted to ordinal, meaning that the categories of each variable will be indicated through a number. The conversion procedure is saved in a dictionary, in case we would like to know the original category of each variable.

# In[ ]:


ordinal = ['Alley','BldgType','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtQual','CentralAir','Condition1','Condition2','Electrical','ExterCond','Exterior1st','Exterior2nd',           'ExterQual', 'Fence','FireplaceQu','Foundation','Functional','GarageCond','GarageFinish','GarageQual','GarageType','Heating','HeatingQC','HouseStyle','KitchenQual','LandContour',           'LandSlope','LotConfig','LotShape','MasVnrArea','MasVnrType','MiscFeature','MSSubClass','MSZoning','Neighborhood','OverallCond','OverallQual','PavedDrive','PoolQC','RoofMatl',           'RoofStyle','SaleCondition','Street','Utilities']


# In[ ]:


def categorify(df, var, d = None):
    df_cp = df.copy(deep=True)
    codebook = d if d else dict()  
    for v in var:
        if v not in codebook.keys():
            df_cp[v] = df_cp[v].astype('category')
            keys = np.sort(df[v].unique())
            if np.array_equal(keys,np.arange(len(keys))):
                assert bool(codebook),"No dictionary provided. Please provide a dictionary to avoid overwriting values"
            else:
                df_cp[v] = df_cp[v].cat.reorder_categories(keys,ordered=True)
                values = df_cp[v].cat.codes
                df_cp[v] = values
                codebook[v] = list(zip(keys,np.arange(len(keys))))
    return df_cp,codebook


# In[ ]:


data,codebook = categorify(data_clean,ordinal)


# ## 3.5 Normalizing numerical values

# Finally, we are normalizing the numerical variables in the dataset to ease the regression fitting process.

# In[ ]:



need_norm = ['1stFlrSF','2ndFlrSF','3SsnPorch','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','EnclosedPorch','GarageArea','MasVnrArea','OpenPorchSF','PoolArea','ScreenPorch','TotalBsmtSF','WoodDeckSF','LotArea','LotFrontage','TotalSF']


# In[ ]:


def normalize(df,need_norm):
    df = df.astype('float64')
    for v in need_norm:
        df[v] = (df[v]-df[v].mean())/df[v].std()
    return df
data_final = normalize(data,need_norm)
data_final


# ## 3.6 Fixing target and creating train, test set.
# 

# Finally, we are going to take a look at the distribution of SalePrice

# In[ ]:


sns.distplot(y_train)


# The prices resemble a left-skewed normal distribution. Since regression models work better with normally distributed data, we are going to apply a logarithmic function to them to normalize them.

# In[ ]:


y_train_log = np.log(y_train)
sns.distplot(y_train_log)


# Splitting data back into train and test set and droppign 'Id' variable:

# In[ ]:


data_final = data
m = train_raw.shape[0]-2
x_train = data_final.loc[:(m-1),:]
x_test = data_final.loc[m:,:]


# In[ ]:


xtrain  = x_train.drop(labels = 'Id', axis = 1)
xtest = x_test.drop(labels = 'Id',axis = 1)


# # 4. Training stacked regressors

# The idea of the model is to use 6 different regression techniques and to average over their result. The algorithms used are:
# * Lasso
# * ElasticNet
# * Ridge Regression
# * Gradient Boosting Regression
# * XGBoost
# * LightGBM

# In[ ]:


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC,Ridge
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


# We use the **cross_val_score** function of Sklearn to create a 5-fold cross validation function that returns the logarithmic Root Mean Squared Error for each model.

# In[ ]:


#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(xtrain)
    rmse= np.sqrt(-cross_val_score(model, xtrain, y_train_log, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# Base Models:

# -  **LASSO  Regression**  : 
# 
# This model may be very sensitive to outliers. So we need to made it more robust on them. For that we use the sklearn's  **Robustscaler()**  method on pipeline.

# In[ ]:


lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))


# - **Elastic Net Regression** :
# 
# again made robust to outliers

# In[ ]:


ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))


# - **Ridge Regression** :
# 
# Ridge regression, made robust against outliers since its performance suffers from them heavily.

# In[ ]:


RR =make_pipeline(RobustScaler(), Ridge(alpha=0.8, random_state=1))


# - **Gradient Boosting Regression** :
# 
# With **huber**  loss that makes it robust to outliers
#     

# In[ ]:


GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)


# - **XGBoost** :

# In[ ]:


model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)


# - **LightGBM** :

# In[ ]:


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


# Model performance on the data by evaluating the  cross-validation rmsle error[](http://)

# In[ ]:


score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(RR)
print("Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))


# ## 4.1 Stacking  models

# We begin with this simple approach of averaging base models.  We build a new **class**  to extend scikit-learn with our model and also to laverage encapsulation and code reuse ([inheritance][1]) 
# 
# 
#   [1]: https://en.wikipedia.org/wiki/Inheritance_(object-oriented_programming)

# In[ ]:


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   


# **Averaged base models score**

# In[ ]:


averaged_models = AveragingModels(models = (ENet, GBoost, RR, lasso,model_xgb,model_lgb))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


get_ipython().run_cell_magic('capture', '', 'averaged_models.fit(xtrain, y_train_log)')


# # Results

# In[ ]:


train_pred = np.exp(averaged_models.predict(xtrain))
test_pred = np.exp(averaged_models.predict(xtest))


# In[ ]:


submission = pd.DataFrame()
submission['Id'] = x_test['Id'].astype('int32')
submission['SalePrice'] = test_pred
submission.to_csv('submission.csv',index=False)

