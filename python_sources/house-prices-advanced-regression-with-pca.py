#!/usr/bin/env python
# coding: utf-8

# # Exploratory Analysis

# In[ ]:


from __future__ import print_function  # Compatability with Python 3

print( 'Print function ready to serve.')


# ### import the libraries we'll need

# In[ ]:


# NumPy for numerical computing
import numpy as np

# Pandas for DataFrames
import pandas as pd
pd.set_option('display.max_columns', 100)

# Matplotlib for visualization
from matplotlib import pyplot as plt
# display plots in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Seaborn for easier visualization
import seaborn as sns

from scipy import stats
sns.set()

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
#warnings.filterwarnings("ignore")

from subprocess import check_output
print(check_output(["ls", "/"]).decode("utf8"))

import os
print(os.listdir("../input"))


# In[ ]:


# Load real estate data from CSV
df_train = pd.read_csv("../input/train.csv", index_col=0)
df_test = pd.read_csv("../input/test.csv", index_col=0)


# In[ ]:


df_test_id_test = pd.read_csv("../input/test.csv")


# In[ ]:


# setting the number of cross validations used in the Model part 
nr_cv = 5

# switch for using log values for SalePrice and features     
use_logvals = 1    
# target used for correlation 
target = 'SalePrice_Log'
    
# only columns with correlation above this threshold value  
# are used for the ML Regressors in Part 3
min_val_corr = 0.4    
    
# switch for dropping columns that are similar to others already used and show a high correlation to these     
drop_similar = 1


# Some useful functions

# In[ ]:


def get_best_score(grid):
    
    best_score = np.sqrt(-grid.best_score_)
    print(best_score)    
    print(grid.best_params_)
    print(grid.best_estimator_)
    
    return best_score


# In[ ]:


def print_cols_large_corr(df, nr_c, targ) :
    corr = df.corr()
    corr_abs = corr.abs()
    print (corr_abs.nlargest(nr_c, targ)[targ])


# In[ ]:


def plot_corr_matrix(df, nr_c, targ) :
    
    corr = df.corr()
    corr_abs = corr.abs()
    cols = corr_abs.nlargest(nr_c, targ)[targ].index
    cm = np.corrcoef(df[cols].values.T)

    plt.figure(figsize=(nr_c/1.5, nr_c/1.5))
    sns.set(font_scale=1.25)
    sns.heatmap(cm, linewidths=1.5, annot=True, square=True, 
                fmt='.2f', annot_kws={'size': 10}, 
                yticklabels=cols.values, xticklabels=cols.values
               )
    plt.show()


# ## Basic information

# ### shape, info, head and describe

# In[ ]:


# Dataframe dimensions
print(df_train.shape)
print("*"*50)
print(df_test.shape)


# In[ ]:


# Column datatypes
print(df_train.dtypes)
print("*"*50)
print(df_test.dtypes)


# In[ ]:


# Type of df.types
type(df_train.dtypes)


# In[ ]:


# Display first 5 rows of df_train
df_train.head()


# In[ ]:


# Display first 5 rows of df_test
df_test.head()


# In[ ]:


# Filter and display only df.dtypes that are 'object'
df_train.dtypes[df_train.dtypes == 'object']


# In[ ]:


# Loop through categorical feature names and print each one
for feature in df_train.dtypes[df_train.dtypes == 'object'].index:
    print(feature)


# In[ ]:


# Display the first 10 rows of data
df_train.head(10)


# In[ ]:


# Display last 5 rows of data
df_train.tail()


# ## Distributions of numeric features

# In[ ]:


# Plot histogram grid
df_train.hist(figsize=(20,20), xrot=-45)

# Clear the text "residue"
plt.show()


# In[ ]:


# Summarize numerical features
df_train.describe()


# In[ ]:


df_test.describe()


# ### The target variable : Distribution of SalePrice

# In[ ]:


sns.distplot(df_train['SalePrice']);
#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())


# the target variable SalePrice is not normally distributed.
# This can reduce the performance of the ML models because they assume normal distribution, see sklearn info on preprocessing
# 
# Therfore we make a log transformation, the resulting distribution looks much better.

# In[ ]:


df_train['SalePrice_Log'] = np.log1p(df_train['SalePrice'])

sns.distplot(df_train['SalePrice_Log']);
# skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice_Log'].skew())
print("Kurtosis: %f" % df_train['SalePrice_Log'].kurt())


# Display summary statistics for categorical features.

# In[ ]:


# Summarize categorical features
df_train.describe(include=['object'])


# Number of Numerical and Categorical features

# In[ ]:


numerical_feats = df_train.dtypes[df_train.dtypes != "object"].index
print("Number of Numerical features: ", len(numerical_feats))

categorical_feats = df_train.dtypes[df_train.dtypes == "object"].index
print("Number of Categorical features: ", len(categorical_feats))


# Columns of Numerical and Categorical features

# In[ ]:


print(df_train[numerical_feats].columns)
print("*"*100)
print(df_train[categorical_feats].columns)


# In[ ]:


# Plot bar plot for each categorical feature

for feature in df_train.dtypes[df_train.dtypes == 'object'].index:
    sns.countplot(y=feature, data=df_train)
    plt.show()


# List of features with missing values

# In[ ]:


total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# Filling missing values
# For a few columns there is lots of NaN entries.
# However, reading the data description we find this is not missing data:
# For PoolQC, NaN is not missing data but means no pool, likewise for Fence, FireplaceQu etc.

# In[ ]:


# columns where NaN values have meaning e.g. no pool etc.
cols_fillna = ['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu',
               'GarageQual','GarageCond','GarageFinish','GarageType', 'Electrical',
               'KitchenQual', 'SaleType', 'Functional', 'Exterior2nd', 'Exterior1st',
               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2',
               'MSZoning', 'Utilities']

# replace 'NaN' with 'None' in these columns
for col in cols_fillna:
    df_train[col].fillna('None',inplace=True)
    df_test[col].fillna('None',inplace=True)


# In[ ]:


total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(5)


# In[ ]:


# fillna with mean or mode for the remaining values
df_train.fillna(df_train.mean(), inplace=True)
df_test.fillna(df_test.mean(), inplace=True)
df_train.fillna(df_train.mode(), inplace=True)
df_test.fillna(df_test.mode(), inplace=True)


# In[ ]:


df_train.isnull().sum().sum()


# In[ ]:


df_test.isnull().sum().sum()


# log transform
# Like the target variable, also some of the feature values are not normally distributed and it is therefore better to use log values in df_train and df_test. Checking for skewness and kurtosis:

# In[ ]:


for col in numerical_feats:
    print(col)
    print("Skewness: %f" % df_train[col].skew())
    print("Kurtosis: %f" % df_train[col].kurt())
    print("*"*50)


# In[ ]:


sns.distplot(df_train['GrLivArea']);
#skewness and kurtosis
print("Skewness: %f" % df_train['GrLivArea'].skew())
print("Kurtosis: %f" % df_train['GrLivArea'].kurt())


# In[ ]:


sns.distplot(df_train['LotArea']);
#skewness and kurtosis
print("Skewness: %f" % df_train['LotArea'].skew())
print("Kurtosis: %f" % df_train['LotArea'].kurt())


# In[ ]:


for df in [df_train, df_test]:
    df['GrLivArea_Log'] = np.log(df['GrLivArea'])
    df.drop('GrLivArea', inplace= True, axis = 1)
    df['LotArea_Log'] = np.log(df['LotArea'])
    df.drop('LotArea', inplace= True, axis = 1)
    
    
    
numerical_feats = df_train.dtypes[df_train.dtypes != "object"].index


# In[ ]:


sns.distplot(df_train['GrLivArea_Log']);
#skewness and kurtosis
print("Skewness: %f" % df_train['GrLivArea_Log'].skew())
print("Kurtosis: %f" % df_train['GrLivArea_Log'].kurt())


# In[ ]:


sns.distplot(df_train['LotArea_Log']);
#skewness and kurtosis
print("Skewness: %f" % df_train['LotArea_Log'].skew())
print("Kurtosis: %f" % df_train['LotArea_Log'].kurt())


# # Relation of features to target (SalePrice_log)

# ### Plots of relation to target for all numerical features

# In[ ]:


nr_rows = 12
nr_cols = 3

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*3.5,nr_rows*3))

li_num_feats = list(numerical_feats)
li_not_plot = ['Id', 'SalePrice', 'SalePrice_Log']
li_plot_num_feats = [c for c in list(numerical_feats) if c not in li_not_plot]


for r in range(0,nr_rows):
    for c in range(0,nr_cols):  
        i = r*nr_cols+c
        if i < len(li_plot_num_feats):
            sns.regplot(df_train[li_plot_num_feats[i]], df_train[target], ax = axs[r][c])
            stp = stats.pearsonr(df_train[li_plot_num_feats[i]], df_train[target])
            #axs[r][c].text(0.4,0.9,"title",fontsize=7)
            str_title = "r = " + "{0:.2f}".format(stp[0]) + "      " "p = " + "{0:.2f}".format(stp[1])
            axs[r][c].set_title(str_title,fontsize=11)
            
plt.tight_layout()    
plt.show()


# Conclusion from EDA on numerical columns:
# 
# We see that for some features like 'OverallQual' there is a strong linear correlation (0.79) to the target.
# For other features like 'MSSubClass' the correlation is very weak.
# For this kernel I decided to use only those features for prediction that have a correlation larger than a threshold value to SalePrice.
# This threshold value can be choosen in the global settings : min_val_corr
# 
# With the default threshold for min_val_corr = 0.4, these features are dropped in Part 2, Data Wrangling:
# 'Id', 'MSSubClass', 'LotArea', 'OverallCond', 'BsmtFinSF2', 'BsmtUnfSF', 'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'HalfBath',
# 'BedroomAbvGr', 'KitchenAbvGr', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'
# 
# We also see that the entries for some of the numerical columns are in fact categorical values.
# For example, the numbers for 'OverallQual' and 'MSSubClass' represent a certain group for that feature ( see data description txt)

# Outliers
# 
# Find columns with strong correlation to target
# Only those with r > min_val_corr are used in the ML Regressors in Part 3
# The value for min_val_corr can be chosen in global settings

# In[ ]:


corr = df_train.corr()
corr_abs = corr.abs()

nr_num_cols = len(numerical_feats)
ser_corr = corr_abs.nlargest(nr_num_cols, target)[target]

cols_abv_corr_limit = list(ser_corr[ser_corr.values > min_val_corr].index)
cols_bel_corr_limit = list(ser_corr[ser_corr.values <= min_val_corr].index)


# List of numerical features and their correlation coefficient to target

# In[ ]:


print(ser_corr)
print("*"*30)
print("List of numerical features with r above min_val_corr :")
print(cols_abv_corr_limit)
print("*"*30)
print("List of numerical features with r below min_val_corr :")
print(cols_bel_corr_limit)


# List of categorical features and their unique values

# In[ ]:


for catg in list(categorical_feats) :
    print(df_train[catg].value_counts())
    print('#'*50)


# Relation to SalePrice for all categorical features

# In[ ]:


li_cat_feats = list(categorical_feats)
nr_rows = 15
nr_cols = 3

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*4,nr_rows*3))

for r in range(0,nr_rows):
    for c in range(0,nr_cols):  
        i = r*nr_cols+c
        if i < len(li_cat_feats):
            sns.boxplot(x=li_cat_feats[i], y=target, data=df_train, ax = axs[r][c])
    
plt.tight_layout()    
plt.show()   


# Conclusion from EDA on categorical columns:
# 
# For many of the categorical there is no strong relation to the target.
# However, for some fetaures it is easy to find a strong relation.
# From the figures above these are : 'MSZoning', 'Neighborhood', 'Condition2', 'MasVnrType', 'ExterQual', 'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType' Also for the categorical features, I use only those that show a strong relation to SalePrice. So the other columns are dropped when creating the ML dataframes in Part 2 :
# 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterCond', 'Foundation', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleCondition'

# In[ ]:


catg_strong_corr = [ 'MSZoning', 'Neighborhood', 'Condition2', 'MasVnrType', 'ExterQual', 
                     'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType']

catg_weak_corr = ['Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 
                  'LandSlope', 'Condition1',  'BldgType', 'HouseStyle', 'RoofStyle', 
                  'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterCond', 'Foundation', 
                  'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 
                  'HeatingQC', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 
                  'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 
                  'SaleCondition' ]
      


# Correlation matrix 1
# <br> Features with largest correlation to SalePrice_Log
# <br> all numerical features with correlation coefficient above threshold

# In[ ]:


nr_feats = len(cols_abv_corr_limit)


# In[ ]:


plot_corr_matrix(df_train, nr_feats, target)


# Of those features with the largest correlation to SalePrice, some also are correlated strongly to each other.
# 
# <br> To avoid failures of the ML regression models due to multicollinearity, these are dropped in part 2.
# 
# <br> This is optional and controlled by the switch drop_similar (global settings)

# ## Part 2: Data wrangling
# <br> Drop all columns with only small correlation to SalePrice
# <br> Transform Categorical to numerical 
# <br> Handling columns with missing data
# <br> Log values
# <br> Drop all columns with strong correlation to similar features
# 
# <br> Numerical columns : drop similar and low correlation
# 
# <br> Categorical columns : Transform to numerical
# 
# <br> Dropping all columns with weak correlation to SalePrice

# In[ ]:


id_test = df_test_id_test['Id']


# In[ ]:




to_drop_num  = cols_bel_corr_limit
to_drop_catg = catg_weak_corr

# cols_to_drop = ['Id'] + to_drop_num + to_drop_catg 

cols_to_drop = to_drop_num + to_drop_catg 

for df in [df_train, df_test]:
    df.drop(cols_to_drop, inplace= True, axis = 1)


# Convert categorical columns to numerical
# <br> For those categorcial features where the EDA with boxplots seem to show a strong dependence of the SalePrice on the 
# <br> category, we transform the columns to numerical. To investigate the relation of the categories to SalePrice in more detail, we 
# <br> make violinplots for these features Also, we look at the mean of SalePrice as function of category.

# In[ ]:


catg_list = catg_strong_corr.copy()
catg_list.remove('Neighborhood')

for catg in catg_list :
    #sns.catplot(x=catg, y=target, data=df_train, kind='boxen')
    sns.violinplot(x=catg, y=target, data=df_train)
    plt.show()
    #sns.boxenplot(x=catg, y=target, data=df_train)
    #bp = df_train.boxplot(column=[target], by=catg)


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(16, 5)
sns.violinplot(x='Neighborhood', y=target, data=df_train, ax=ax)
plt.xticks(rotation=45)
plt.show()


# In[ ]:


for catg in catg_list :
    g = df_train.groupby(catg)[target].mean()
    print(g)


# In[ ]:


# 'MSZoning'
msz_catg2 = ['RM', 'RH']
msz_catg3 = ['RL', 'FV'] 


# Neighborhood
nbhd_catg2 = ['Blmngtn', 'ClearCr', 'CollgCr', 'Crawfor', 'Gilbert', 'NWAmes', 'Somerst', 'Timber', 'Veenker']
nbhd_catg3 = ['NoRidge', 'NridgHt', 'StoneBr']

# Condition2
cond2_catg2 = ['Norm', 'RRAe']
cond2_catg3 = ['PosA', 'PosN'] 

# SaleType
SlTy_catg1 = ['Oth']
SlTy_catg3 = ['CWD']
SlTy_catg4 = ['New', 'Con']

#[]


# In[ ]:


for df in [df_train, df_test]:
    
    df['MSZ_num'] = 1  
    df.loc[(df['MSZoning'].isin(msz_catg2) ), 'MSZ_num'] = 2    
    df.loc[(df['MSZoning'].isin(msz_catg3) ), 'MSZ_num'] = 3        
    
    df['NbHd_num'] = 1       
    df.loc[(df['Neighborhood'].isin(nbhd_catg2) ), 'NbHd_num'] = 2    
    df.loc[(df['Neighborhood'].isin(nbhd_catg3) ), 'NbHd_num'] = 3    

    df['Cond2_num'] = 1       
    df.loc[(df['Condition2'].isin(cond2_catg2) ), 'Cond2_num'] = 2    
    df.loc[(df['Condition2'].isin(cond2_catg3) ), 'Cond2_num'] = 3    
    
    df['Mas_num'] = 1       
    df.loc[(df['MasVnrType'] == 'Stone' ), 'Mas_num'] = 2 
    
    df['ExtQ_num'] = 1       
    df.loc[(df['ExterQual'] == 'TA' ), 'ExtQ_num'] = 2     
    df.loc[(df['ExterQual'] == 'Gd' ), 'ExtQ_num'] = 3     
    df.loc[(df['ExterQual'] == 'Ex' ), 'ExtQ_num'] = 4     
   
    df['BsQ_num'] = 1          
    df.loc[(df['BsmtQual'] == 'Gd' ), 'BsQ_num'] = 2     
    df.loc[(df['BsmtQual'] == 'Ex' ), 'BsQ_num'] = 3     
 
    df['CA_num'] = 0          
    df.loc[(df['CentralAir'] == 'Y' ), 'CA_num'] = 1    

    df['Elc_num'] = 1       
    df.loc[(df['Electrical'] == 'SBrkr' ), 'Elc_num'] = 2 


    df['KiQ_num'] = 1       
    df.loc[(df['KitchenQual'] == 'TA' ), 'KiQ_num'] = 2     
    df.loc[(df['KitchenQual'] == 'Gd' ), 'KiQ_num'] = 3     
    df.loc[(df['KitchenQual'] == 'Ex' ), 'KiQ_num'] = 4      
    
    df['SlTy_num'] = 2       
    df.loc[(df['SaleType'].isin(SlTy_catg1) ), 'SlTy_num'] = 1  
    df.loc[(df['SaleType'].isin(SlTy_catg3) ), 'SlTy_num'] = 3  
    df.loc[(df['SaleType'].isin(SlTy_catg4) ), 'SlTy_num'] = 4  


# Checking correlation to SalePrice for the new numerical columns

# In[ ]:


new_col_num = ['MSZ_num', 'NbHd_num', 'Cond2_num', 'Mas_num', 'ExtQ_num', 'BsQ_num', 'CA_num', 'Elc_num', 'KiQ_num', 'SlTy_num']

nr_rows = 4
nr_cols = 3

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*3.5,nr_rows*3))

for r in range(0,nr_rows):
    for c in range(0,nr_cols):  
        i = r*nr_cols+c
        if i < len(new_col_num):
            sns.regplot(df_train[new_col_num[i]], df_train[target], ax = axs[r][c])
            stp = stats.pearsonr(df_train[new_col_num[i]], df_train[target])
            str_title = "r = " + "{0:.2f}".format(stp[0]) + "      " "p = " + "{0:.2f}".format(stp[1])
            axs[r][c].set_title(str_title,fontsize=11)
            
plt.tight_layout()    
plt.show()   


# There are few columns with quite large correlation to SalePrice (NbHd_num, ExtQ_num, BsQ_num, KiQ_num).
# <br> These will probably be useful for optimal performance of the Regressors in part 3.
# 
# <br> Dropping the converted categorical columns and the new numerical columns with weak correlation
# 
# <br> columns and correlation before dropping
# 
# 

# In[ ]:


catg_cols_to_drop = ['Neighborhood' , 'Condition2', 'MasVnrType', 'ExterQual', 'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType']

corr1 = df_train.corr()
corr_abs_1 = corr1.abs()

nr_all_cols = len(df_train)
ser_corr_1 = corr_abs_1.nlargest(nr_all_cols, target)[target]

print(ser_corr_1)
cols_bel_corr_limit_1 = list(ser_corr_1[ser_corr_1.values <= min_val_corr].index)


for df in [df_train, df_test] :
    df.drop(catg_cols_to_drop, inplace= True, axis = 1)
    df.drop(cols_bel_corr_limit_1, inplace= True, axis = 1)    


# columns and correlation after dropping

# In[ ]:


corr2 = df_train.corr()
corr_abs_2 = corr2.abs()

nr_all_cols = len(df_train)
ser_corr_2 = corr_abs_2.nlargest(nr_all_cols, target)[target]

print(ser_corr_2)


# new dataframes

# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# ### List of all features with strong correlation to SalePrice_Log
# <br> after dropping all coumns with weak correlation

# In[ ]:


corr = df_train.corr()
corr_abs = corr.abs()

nr_all_cols = len(df_train)
print (corr_abs.nlargest(nr_all_cols, target)[target])


# #### Correlation Matrix 2 : All features with strong correlation to SalePrice

# In[ ]:


nr_feats=len(df_train.columns)
plot_corr_matrix(df_train, nr_feats, target)


# Check for Multicollinearity
# 
# <br> Strong correlation of these features to other, similar features:
# 
# <br> 'GrLivArea_Log' and 'TotRmsAbvGrd'
# 
# <br> 'GarageCars' and 'GarageArea'
# 
# <br> 'TotalBsmtSF' and '1stFlrSF'
# 
# <br> 'YearBuilt' and 'GarageYrBlt'
# 
# <br> Of those features we drop the one that has smaller correlation coeffiecient to Target.

# In[ ]:


cols = corr_abs.nlargest(nr_all_cols, target)[target].index
cols = list(cols)

if drop_similar == 1 :
    for col in ['GarageArea','1stFlrSF','TotRmsAbvGrd','GarageYrBlt'] :
        if col in cols: 
            cols.remove(col)


# In[ ]:


cols = list(cols)
print(cols)


# #### List of features used for the Regressors in Part 3

# In[ ]:


feats = cols.copy()
feats.remove('SalePrice_Log')
feats.remove('SalePrice')

print(feats)


# In[ ]:


df_test.head()


# In[ ]:


df_train_ml = df_train[feats].copy()
df_test_ml  = df_test[feats].copy()

y = df_train[target]

print(target)


# In[ ]:


# Dataframe dimensions
print(df_train.shape)
print("*"*50)
print(df_test.shape)
print("*"*50)
print(df_train_ml.shape)
print("*"*50)
print(df_test_ml.shape)


# In[ ]:


numerical_feats = df_train_ml.dtypes[df_train_ml.dtypes != "object"].index
print("Number of Numerical features: ", len(numerical_feats))

categorical_feats = df_train_ml.dtypes[df_train_ml.dtypes == "object"].index
print("Number of Categorical features: ", len(categorical_feats))


# In[ ]:


df_all = pd.concat([df_train_ml, df_test_ml])

len_train = df_train_ml.shape[0]

xtrain = df_all[:len_train]
xtest = df_all[len_train:]


# In[ ]:


# Dataframe dimensions
print(xtrain.shape)
print("*"*50)
print(xtest.shape)


# In[ ]:


xtrain.head()


# In[ ]:


xtest.head()


# In[ ]:


# StandardScaler from Scikit-Learn
from sklearn.preprocessing import StandardScaler

# Initialize instance of StandardScaler
scaler = StandardScaler()

# Fit and transform item_data
data_scaled = scaler.fit_transform(df_all)

# Display first 5 rows of item_data_scaled
data_scaled[:5]


# In[ ]:


# Plot scatterplot of scaled x1 against scaled x2
plt.scatter(data_scaled[:,0], data_scaled[:,1])

# Put plot axes on the same scale
plt.axis('equal')

# Label axes
plt.xlabel('x1 (scaled)')
plt.ylabel('x2 (scaled)')

# Clear text residue
plt.show()


# In[ ]:


total = df_all.isnull().sum().sort_values(ascending=False)
percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[ ]:


# Initialize instance of PCA transformation
from sklearn.decomposition import PCA
pca = PCA()

# Fit the instance
pca.fit(data_scaled)


# In[ ]:


# Display principal components
pca.components_


# In[ ]:


# Plot scaled dataset and make it partially transparent
plt.scatter(data_scaled[:,0], data_scaled[:,1], alpha=0.3)

# Plot first principal component in black
plt.plot([0, 2*pca.components_[0,0]], [0, 2*pca.components_[0,1]], 'k')

# Plot second principal component in red
plt.plot([0, pca.components_[1,0]], [0, pca.components_[1,1]], 'r')

# Set axes
plt.axis('equal')
plt.xlabel('x1 (scaled)')
plt.ylabel('x2 (scaled)')

# Clear text residue
plt.show()


# In[ ]:


# Generate new features
PC = pca.transform(data_scaled)

# Display first 5 rows
PC[:5]


# In[ ]:


# Cumulative explained variance
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot cumulative explained variance
plt.plot(range(len(cumulative_explained_variance)), cumulative_explained_variance)


# In[ ]:


# How much variance we'd capture with the first 8 components
cumulative_explained_variance[8]


# In[ ]:


# Initialize PCA transformation, only keeping 8 components
pca = PCA(n_components=8)

# Fit and transform item_data_scaled
PC_items = pca.fit_transform(data_scaled)

# Display shape of PC_items
PC_items.shape


# In[ ]:


# Put PC_items into a dataframe
items_pca = pd.DataFrame(PC_items)

# Name the columns
items_pca.columns = ['PC{}'.format(i + 1) for i in range(PC_items.shape[1])]

# Update its index
items_pca.index = df_all.index

# Display first 5 rows
items_pca.head()


# In[ ]:


len_train = df_train_ml.shape[0]

pca_xtrain = items_pca[:len_train]
pca_xtest = items_pca[len_train:]


# In[ ]:


# Dataframe dimensions
print(pca_xtrain.shape)
print("*"*50)
print(pca_xtest.shape)


# In[ ]:


# Plot transformed dataset
plt.scatter(PC[:,0], PC[:,1], alpha=0.3, color='g')

# Plot first principal component in black
plt.plot([0, 2], [0, 0], 'k')

# Plot second principal component in red
plt.plot([0, 0], [0, 1], 'r')

# Set axes
plt.axis('equal')
plt.xlabel('PC1')
plt.ylabel('PC2')

# Clear text residue
plt.show()


# In[ ]:


# Display explained variance ratio
pca.explained_variance_ratio_


# ##### import the libraries we'll need

# In[ ]:


# NumPy for numerical computing
import numpy as np

# Pandas for DataFrames
import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Matplotlib for visualization
from matplotlib import pyplot as plt
# display plots in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Seaborn for easier visualization
import seaborn as sns

# Scikit-Learn for Modeling
import sklearn

# Import Elastic Net, Ridge Regression, and Lasso Regression
from sklearn.linear_model import ElasticNet, Ridge, Lasso

# Import Random Forest and Gradient Boosted Trees
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# Creating Datasets for ML algorithms

# In[ ]:


X = pca_xtrain.copy()
y = df_train[target]
X_test = pca_xtest.copy()
#y_test = df_train[target]

X.info()
X_test.info()


# In[ ]:


# Split X and y into train and test sets
X_train = pca_xtrain.copy()
X_test = pca_xtest.copy()
y_train = df_train[target]
#y_test = df_test[target]


# In[ ]:


X.head()


# In[ ]:


X_test.head()


# In[ ]:


print( len(X_train), len(X_test), len(y_train) )


# In[ ]:


# Summary statistics of X_train
X_train.describe()


# In[ ]:


# Standardize X_train
X_train_new = (X_train - X_train.mean()) / X_train.std()


# In[ ]:


# Summary statistics of X_train_new
X_train_new.describe()


# In[ ]:


# Function for creating model pipelines
from sklearn.pipeline import make_pipeline


# In[ ]:


# For standardization
from sklearn.preprocessing import StandardScaler


# In[ ]:


make_pipeline(StandardScaler(), Lasso(random_state=123))


# In[ ]:


# Create pipelines dictionary
pipelines = {
    'lasso' : make_pipeline(StandardScaler(), Lasso(random_state=123)),
    'ridge' : make_pipeline(StandardScaler(), Ridge(random_state=123)),
    'enet'  : make_pipeline(StandardScaler(), ElasticNet(random_state=123))
}


# In[ ]:


# Add a pipeline for 'rf'
pipelines['rf'] = make_pipeline(StandardScaler(), RandomForestRegressor(random_state=123))

# Add a pipeline for 'gb'
pipelines['gb'] = make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=123))


# In[ ]:


# Check that we have all 5 algorithms, and that they are all pipelines
for key, value in pipelines.items():
    print( key, type(value) )


# In[ ]:


# List tuneable hyperparameters of our Lasso pipeline
pipelines['lasso'].get_params()


# In[ ]:


# Lasso hyperparameters
lasso_hyperparameters = { 
    'lasso__alpha' : [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10] 
}

# Ridge hyperparameters
ridge_hyperparameters = { 
    'ridge__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]  
}


# In[ ]:


# Elastic Net hyperparameters
enet_hyperparameters = { 
    'elasticnet__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],                        
    'elasticnet__l1_ratio' : [0.1, 0.3, 0.5, 0.7, 0.9]  
}


# In[ ]:


# Random forest hyperparameters
rf_hyperparameters = { 
    'randomforestregressor__n_estimators' : [100, 200],
    'randomforestregressor__max_features': ['auto', 'sqrt', 0.33],
}


# In[ ]:


# Boosted tree hyperparameters
gb_hyperparameters = { 
    'gradientboostingregressor__n_estimators': [100, 200],
    'gradientboostingregressor__learning_rate' : [0.05, 0.1, 0.2],
    'gradientboostingregressor__max_depth': [1, 3, 5]
}


# In[ ]:


# Create hyperparameters dictionary
hyperparameters = {
    'rf' : rf_hyperparameters,
    'gb' : gb_hyperparameters,
    'lasso' : lasso_hyperparameters,
    'ridge' : ridge_hyperparameters,
    'enet' : enet_hyperparameters
}


# In[ ]:


for key in ['enet', 'gb', 'ridge', 'rf', 'lasso']:
    if key in hyperparameters:
        if type(hyperparameters[key]) is dict:
            print( key, 'was found in hyperparameters, and it is a grid.' )
        else:
            print( key, 'was found in hyperparameters, but it is not a grid.' )
    else:
        print( key, 'was not found in hyperparameters')


# In[ ]:


# Helper for cross-validation
from sklearn.model_selection import GridSearchCV


# In[ ]:


# Create cross-validation object from Lasso pipeline and Lasso hyperparameters
model = GridSearchCV(pipelines['lasso'], hyperparameters['lasso'], cv=10, n_jobs=-1)


# In[ ]:


type(model)


# In[ ]:


# Fit and tune model
model.fit(X_train, y_train)


# In[ ]:


# Create empty dictionary called fitted_models
fitted_models = {}

# Loop through model pipelines, tuning each one and saving it to fitted_models
for name, pipeline in pipelines.items():
    # Create cross-validation object from pipeline and hyperparameters
    model = GridSearchCV(pipeline, hyperparameters[name], cv=10, n_jobs=-1)
    
    # Fit model on X_train, y_train
    model.fit(X_train, y_train)
    
    # Store model in fitted_models[name] 
    fitted_models[name] = model
    
    # Print '{name} has been fitted'
    print(name, 'has been fitted.')


# In[ ]:


# Check that we have 5 cross-validation objects
for key, value in fitted_models.items():
    print( key, type(value) )


# In[ ]:


from sklearn.exceptions import NotFittedError

for name, model in fitted_models.items():
    try:
        pred = model.predict(X_test)
        print(name, 'has been fitted.')
    except NotFittedError as e:
        print(repr(e))


# In[ ]:


# Display best_score_ for each fitted model
for name, model in fitted_models.items():
    print( name, model.best_score_ )


# In[ ]:


# Import r2_score and mean_absolute_error functions
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


# In[ ]:


# Display fitted random forest object
fitted_models['rf']


# In[ ]:


# Predict test set using fitted random forest
pred = fitted_models['rf'].predict(X_test)


# In[ ]:


# Calculate and print R^2 and MAE, for training data set
pred_train = fitted_models['rf'].predict(X_train)
print( 'R^2:', r2_score(y_train, pred_train ))
print( 'MAE:', mean_absolute_error(y_train, pred_train))


# In[ ]:


print(pred)


# In[ ]:


df_test.head()


# In[ ]:


# Create final table

pred_pd = pd.DataFrame()
pred_pd['Id'] = id_test
pred_pd['SalePrice'] = np.exp(pred)

pred_pd.head
pred_pd.to_csv('submission_mrig_PCA.csv',index=False)


# In[ ]:




