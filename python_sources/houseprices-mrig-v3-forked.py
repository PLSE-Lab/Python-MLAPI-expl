#!/usr/bin/env python
# coding: utf-8

# # Part 0 : Imports, Settings, Functions

# **Imports**

# In[ ]:


import numpy as np
import pandas as pd
pd.set_option('max_columns', 105)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
#warnings.filterwarnings("ignore")

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))


# **Settings and switches**
# 
# **Here one can choose settings for optimal performance and runtime.**  
# **For example, nr_cv sets the number of cross validations used in GridsearchCV, and**  
# **min_val_corr is the minimum value for the correlation coefficient to the target (only features with larger correlation will be used).** 

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


# **Some useful functions**

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


# **Load data**

# In[ ]:


df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")

#df_train = pd.read_csv("../input/train.csv")
#df_test = pd.read_csv("../input/test.csv")


# # Part 1: Exploratory Data Analysis

# ## 1.1 Overview of features and relation to target
# 
# Let's get a first overview of the train and test dataset  
# How many rows and columns are there?  
# What are the names of the features (columns)?  
# Which features are numerical, which are categorical?  
# How many values are missing?  
# The **shape** and **info** methods answer these questions  
# **head** displays some rows of the dataset  
# **describe** gives a summary of the statistics (only for numerical columns)

# ### shape, info, head and describe

# In[ ]:


print(df_train.shape)
print("*"*50)
print(df_test.shape)


# In[ ]:


print(df_train.info())
print("*"*50)
print(df_test.info())


# df_train has 81 columns (79 features + id and target SalePrice) and 1460 entries (number of rows or house sales)  
# df_test has 80 columns (79 features + id) and 1459 entries  
# There is lots of info that is probably related to the SalePrice like the area, the neighborhood, the condition and quality. 
# Maybe other features are not so important for predicting the target, also there might be a strong correlation for some of the features (like GarageCars and GarageArea).
# For some columns many values are missing: only 7 values for Pool QC in df_train and 3 in df_test

# In[ ]:


df_train.head()


# In[ ]:


df_train.describe()


# In[ ]:


df_test.head()


# In[ ]:


df_test.describe()


# ### The target variable : Distribution of SalePrice

# In[ ]:


sns.distplot(df_train['SalePrice']);
#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())


# As we see, the target variable SalePrice is not normally distributed.  
# This can reduce the performance of the ML models because they assume normal distribution, see [sklearn info on preprocessing](http://scikit-learn.org/stable/modules/preprocessing.html)
# 
# Therfore we make a log transformation, the resulting distribution looks much better.  

# In[ ]:


df_train['SalePrice_Log'] = np.log1p(df_train['SalePrice'])

sns.distplot(df_train['SalePrice_Log']);
# skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice_Log'].skew())
print("Kurtosis: %f" % df_train['SalePrice_Log'].kurt())
# dropping old column
df_train.drop('SalePrice', axis= 1, inplace=True)


# ### Numerical and Categorical features

# In[ ]:


numerical_feats = df_train.dtypes[df_train.dtypes != "object"].index
print("Number of Numerical features: ", len(numerical_feats))

categorical_feats = df_train.dtypes[df_train.dtypes == "object"].index
print("Number of Categorical features: ", len(categorical_feats))


# In[ ]:


print(df_train[numerical_feats].columns)
print("*"*100)
print(df_train[categorical_feats].columns)


# In[ ]:


df_train[numerical_feats].head()


# In[ ]:


df_train[categorical_feats].head()


# ### List of features with missing values

# In[ ]:


total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# **Filling missing values**  
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


# **Missing values in train data ?**

# In[ ]:


df_train.isnull().sum().sum()


# **Missing values in test data ?**

# In[ ]:


df_test.isnull().sum().sum()


# In[ ]:





# ### log transform
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


# ## 1.2 Relation of features to target (SalePrice_log)

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


# **Conclusion from EDA on numerical columns:**
# 
# We see that for some features like 'OverallQual' there is a strong linear correlation (0.79) to the target.  
# For other features like 'MSSubClass' the correlation is very weak.  
# For this kernel I decided to use only those features for prediction that have a correlation larger than a threshold value to SalePrice.  
# This threshold value can be choosen in the global settings : min_val_corr  
# 
# With the default threshold for min_val_corr = 0.4, these features are dropped in Part 2, Data Wrangling:  
# 'Id', 'MSSubClass', 'LotArea', 'OverallCond', 'BsmtFinSF2', 'BsmtUnfSF',  'LowQualFinSF',  'BsmtFullBath', 'BsmtHalfBath', 'HalfBath',   
# 'BedroomAbvGr', 'KitchenAbvGr', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'
# 
# We also see that the entries for some of the numerical columns are in fact categorical values.  
# For example, the numbers for 'OverallQual' and 'MSSubClass' represent a certain group for that feature ( see data description txt)

# **Outliers**

# In[ ]:





# In[ ]:





# **Find columns with strong correlation to target**  
# Only those with r > min_val_corr are used in the ML Regressors in Part 3  
# The value for min_val_corr can be chosen in global settings

# In[ ]:


corr = df_train.corr()
corr_abs = corr.abs()

nr_num_cols = len(numerical_feats)
ser_corr = corr_abs.nlargest(nr_num_cols, target)[target]

cols_abv_corr_limit = list(ser_corr[ser_corr.values > min_val_corr].index)
cols_bel_corr_limit = list(ser_corr[ser_corr.values <= min_val_corr].index)


# ### List of numerical features and their correlation coefficient to target

# In[ ]:


print(ser_corr)
print("*"*30)
print("List of numerical features with r above min_val_corr :")
print(cols_abv_corr_limit)
print("*"*30)
print("List of numerical features with r below min_val_corr :")
print(cols_bel_corr_limit)


# ### List of categorical features and their unique values

# In[ ]:


for catg in list(categorical_feats) :
    print(df_train[catg].value_counts())
    print('#'*50)


# ### Relation to SalePrice for all categorical features

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


# **Conclusion from EDA on categorical columns:**
# 
# For many of the categorical there is no strong relation to the target.  
# However, for some fetaures it is easy to find a strong relation.  
# From the figures above these are : 'MSZoning', 'Neighborhood', 'Condition2', 'MasVnrType', 'ExterQual', 'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType'
# Also for the categorical features, I use only those that show a strong relation to SalePrice. 
# So the other columns are dropped when creating the ML dataframes in Part 2 :  
#  'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1',  'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
# 'Exterior1st', 'Exterior2nd', 'ExterCond', 'Foundation', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 
# 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleCondition' 
#  

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
      


# ### Correlation matrix 1
# **Features with largest correlation to SalePrice_Log**  
# all numerical features with correlation coefficient above threshold 

# In[ ]:


nr_feats = len(cols_abv_corr_limit)


# In[ ]:


plot_corr_matrix(df_train, nr_feats, target)


# **Of those features with the largest correlation to SalePrice, some also are correlated strongly to each other.**
# 
# 
# **To avoid failures of the ML regression models due to multicollinearity, these are dropped in part 2.**
# 
# 
# **This is optional and controlled by the switch drop_similar (global settings)**

# In[ ]:





# ## Part 2: Data wrangling
# 
# **Drop all columns with only small correlation to SalePrice**  
# **Transform Categorical to numerical **  
# **Handling columns with missing data**  
# **Log values**  
# **Drop all columns with strong correlation to similar features**  

# Numerical columns : drop similar and low correlation
# 
# Categorical columns : Transform  to numerical

# In[ ]:





# ### Dropping all columns with weak correlation to SalePrice

# In[ ]:


id_test = df_test['Id']

to_drop_num  = cols_bel_corr_limit
to_drop_catg = catg_weak_corr

cols_to_drop = ['Id'] + to_drop_num + to_drop_catg 

for df in [df_train, df_test]:
    df.drop(cols_to_drop, inplace= True, axis = 1)


# ### Convert categorical columns to numerical  
# For those categorcial features where the EDA with boxplots seem to show a strong dependence of the SalePrice on the category, we transform the columns to numerical.
# To investigate the relation of the categories to SalePrice in more detail, we make violinplots for these features 
# Also, we look at the mean of SalePrice as function of category.

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
  


# In[ ]:





# ### Checking correlation to SalePrice for the new numerical columns

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
# These will probably be useful for optimal performance of the Regressors in part 3.

# **Dropping the converted categorical columns and the new numerical columns with weak correlation**

# **columns and correlation before dropping**

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


# **columns and correlation after dropping**

# In[ ]:


corr2 = df_train.corr()
corr_abs_2 = corr2.abs()

nr_all_cols = len(df_train)
ser_corr_2 = corr_abs_2.nlargest(nr_all_cols, target)[target]

print(ser_corr_2)


# **new dataframes**

# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:





# **List of all features with strong correlation to SalePrice_Log**  
# after dropping all coumns with weak correlation

# In[ ]:


corr = df_train.corr()
corr_abs = corr.abs()

nr_all_cols = len(df_train)
print (corr_abs.nlargest(nr_all_cols, target)[target])


# ### Correlation Matrix 2 : All features with strong correlation to SalePrice

# In[ ]:


nr_feats=len(df_train.columns)
plot_corr_matrix(df_train, nr_feats, target)


# **Check for Multicollinearity**
# 
# Strong correlation of these features to other, similar features:
# 
# 'GrLivArea_Log' and 'TotRmsAbvGrd'
# 
# 'GarageCars' and 'GarageArea'
# 
# 'TotalBsmtSF' and '1stFlrSF'
# 
# 'YearBuilt' and 'GarageYrBlt'
# 
# **Of those features we drop the one that has smaller correlation coeffiecient to Target.**

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


# In[ ]:





# **List of features used for the Regressors in Part 3**

# In[ ]:


feats = cols.copy()
feats.remove('SalePrice_Log')

print(feats)


# In[ ]:





# In[ ]:


df_train_ml = df_train[feats].copy()
df_test_ml  = df_test[feats].copy()

y = df_train[target]


# **Combine train and test data**  
# for one hot encoding (use pandas get dummies) of all categorical features  
# uncommenting the following cell increases the number of features  
# up to now, all models in Part 3 are optimized for not applying one hot encoder  
# when applied, GridSearchCV needs to be rerun

# In[ ]:


"""
all_data = pd.concat((df_train[feats], df_test[feats]))

li_get_dummies = ['OverallQual', 'NbHd_num', 'GarageCars','ExtQ_num', 'KiQ_num',
                  'BsQ_num', 'FullBath', 'Fireplaces', 'MSZ_num']
all_data = pd.get_dummies(all_data, columns=li_get_dummies, drop_first=True)

df_train_ml = all_data[:df_train.shape[0]]
df_test_ml  = all_data[df_train.shape[0]:]
"""


# In[ ]:





# ### StandardScaler

# In[ ]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
df_train_ml_sc = sc.fit_transform(df_train_ml)
df_test_ml_sc = sc.transform(df_test_ml)


# In[ ]:


df_train_ml_sc = pd.DataFrame(df_train_ml_sc)
df_train_ml_sc.head()


# **Creating Datasets for ML algorithms**

# In[ ]:


X = df_train_ml.copy()
y = df_train[target]
X_test = df_test_ml.copy()

X_sc = df_train_ml_sc.copy()
y_sc = df_train[target]
X_test_sc = df_test_ml_sc.copy()

X.info()
X_test.info()


# In[ ]:


X.head()


# In[ ]:


X_sc.head()


# In[ ]:


X_test.head()


# In[ ]:





# ## Part 3: Scikit-learn basic regression models and comparison of results
# 
# **Test simple sklearn models and compare by metrics**
# 
# **We test the following Regressors from scikit-learn:**  
# LinearRegression  
# Ridge  
# Lasso  
# Elastic Net  
# Stochastic Gradient Descent  
# DecisionTreeRegressor  
# RandomForestRegressor  
# SVR 

# **Model tuning and selection with GridSearchCV**

# In[ ]:


from sklearn.model_selection import GridSearchCV
score_calc = 'neg_mean_squared_error'


# ### Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
grid_linear = GridSearchCV(linreg, parameters, cv=nr_cv, verbose=1 , scoring = score_calc)
grid_linear.fit(X, y)

sc_linear = get_best_score(grid_linear)


# In[ ]:


linregr_all = LinearRegression()
#linregr_all.fit(X_train_all, y_train_all)
linregr_all.fit(X, y)
pred_linreg_all = linregr_all.predict(X_test)
pred_linreg_all[pred_linreg_all < 0] = pred_linreg_all.mean()


# In[ ]:


sub_linreg = pd.DataFrame()
sub_linreg['Id'] = id_test
sub_linreg['SalePrice'] = pred_linreg_all
#sub_linreg.to_csv('linreg.csv',index=False)


# ### Ridge

# In[ ]:


from sklearn.linear_model import Ridge

ridge = Ridge()
parameters = {'alpha':[0.001,0.005,0.01,0.1,0.5,1], 'normalize':[True,False], 'tol':[1e-06,5e-06,1e-05,5e-05]}
grid_ridge = GridSearchCV(ridge, parameters, cv=nr_cv, verbose=1, scoring = score_calc)
grid_ridge.fit(X, y)

sc_ridge = get_best_score(grid_ridge)


# In[ ]:


pred_ridge_all = grid_ridge.predict(X_test)


# ### Lasso

# In[ ]:


from sklearn.linear_model import Lasso

lasso = Lasso()
parameters = {'alpha':[1e-03,0.01,0.1,0.5,0.8,1], 'normalize':[True,False], 'tol':[1e-06,1e-05,5e-05,1e-04,5e-04,1e-03]}
grid_lasso = GridSearchCV(lasso, parameters, cv=nr_cv, verbose=1, scoring = score_calc)
grid_lasso.fit(X, y)

sc_lasso = get_best_score(grid_lasso)

pred_lasso = grid_lasso.predict(X_test)


# ### Elastic Net

# In[ ]:


from sklearn.linear_model import ElasticNet

enet = ElasticNet()
parameters = {'alpha' :[0.1,1.0,10], 'max_iter' :[1000000], 'l1_ratio':[0.04,0.05], 
              'fit_intercept' : [False,True], 'normalize':[True,False], 'tol':[1e-02,1e-03,1e-04]}
grid_enet = GridSearchCV(enet, parameters, cv=nr_cv, verbose=1, scoring = score_calc)
grid_enet.fit(X_sc, y_sc)

sc_enet = get_best_score(grid_enet)

pred_enet = grid_enet.predict(X_test_sc)


# ### SGDRegressor  
# Linear model fitted by minimizing a regularized empirical loss with SGD. SGD stands for Stochastic Gradient Descent: the gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule (aka learning rate). The regularizer is a penalty added to the loss function that shrinks model parameters towards the zero vector using either the squared euclidean norm L2 or the absolute norm L1 or a combination of both (Elastic Net). 

# In[ ]:


from sklearn.linear_model import SGDRegressor

sgd = SGDRegressor()
parameters = {'max_iter' :[10000], 'alpha':[1e-05], 'epsilon':[1e-02], 'fit_intercept' : [True]  }
grid_sgd = GridSearchCV(sgd, parameters, cv=nr_cv, verbose=1, scoring = score_calc)
grid_sgd.fit(X_sc, y_sc)

sc_sgd = get_best_score(grid_sgd)

pred_sgd = grid_sgd.predict(X_test_sc)


# ### DecisionTreeRegressor

# In[ ]:


from sklearn.tree import DecisionTreeRegressor

param_grid = { 'max_depth' : [7,8,9,10] , 'max_features' : [11,12,13,14] ,
               'max_leaf_nodes' : [None, 12,15,18,20] ,'min_samples_split' : [20,25,30],
                'presort': [False,True] , 'random_state': [5] }
            
grid_dtree = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=nr_cv, refit=True, verbose=1, scoring = score_calc)
grid_dtree.fit(X, y)

sc_dtree = get_best_score(grid_dtree)

pred_dtree = grid_dtree.predict(X_test)


# In[ ]:


dtree_pred = grid_dtree.predict(X_test)
sub_dtree = pd.DataFrame()
sub_dtree['Id'] = id_test
sub_dtree['SalePrice'] = dtree_pred
#sub_dtree.to_csv('dtreeregr.csv',index=False)


# In[ ]:





# ### RandomForestRegressor

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

param_grid = {'min_samples_split' : [3,4,6,10], 'n_estimators' : [70,100], 'random_state': [5] }
grid_rf = GridSearchCV(RandomForestRegressor(), param_grid, cv=nr_cv, refit=True, verbose=1, scoring = score_calc)
grid_rf.fit(X, y)

sc_rf = get_best_score(grid_rf)


# In[ ]:


pred_rf = grid_rf.predict(X_test)

sub_rf = pd.DataFrame()
sub_rf['Id'] = id_test
sub_rf['SalePrice'] = pred_rf 

if use_logvals == 1:
    sub_rf['SalePrice'] = np.exp(sub_rf['SalePrice']) 

sub_rf.to_csv('rf.csv',index=False)


# In[ ]:


sub_rf.head(10)


# **SVR**

# In[ ]:


if use_logvals == 0 :

    from sklearn.svm import SVR
    from sklearn.model_selection import GridSearchCV
    
    param_grid = {'C': [100000,300000,400000,500000,600000,800000], 'gamma': [0.1, 0.07, 0.05, 0.04, 0.03, 0.02, 0.01, 0.007,0.005], 'kernel': ['rbf']}

    grid_svr = GridSearchCV(SVR(), param_grid, cv=nr_cv, refit=True, verbose=3, scoring = score_calc)
    grid_svr.fit(X_sc, y_sc)

    print(grid_svr.best_score_)
    print(grid_svr.best_params_)
    print(grid_svr.best_estimator_)
    
    svr_pred = grid_svr.predict(X_test_sc)

    sub_svr = pd.DataFrame()
    sub_svr['Id'] = id_test
    sub_svr['SalePrice'] = svr_pred 
    if use_logvals == 1:
        sub_svr['SalePrice'] = np.exp(sub_svr['SalePrice']) 
    
    sub_svr.head()
    sub_svr.to_csv('svr.csv',index=False)
    


# In[ ]:





# In[ ]:





# ### Comparison plot: RMSE of all models

# In[ ]:


list_scores = [sc_linear, sc_ridge, sc_lasso, sc_enet, sc_sgd, sc_dtree, sc_rf]
list_regressors = ['Linear','Ridge','Lasso','ElaNet','SGD','DTr','RF']


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(10,7)
sns.barplot(x=list_regressors, y=list_scores, ax=ax)
plt.ylabel('RMSE')
plt.show()


# In[ ]:





# **correlation of model results**

# In[ ]:


predictions = {'Linear': pred_linreg_all, 'Ridge': pred_ridge_all, 'Lasso': pred_lasso,
               'ElaNet': pred_enet, 'SGD': pred_sgd, 'DTr': pred_dtree, 'RF': pred_rf
              }
df_predictions = pd.DataFrame(data=predictions) 
df_predictions.corr()


# **mean of best models**

# In[ ]:


# Create final table

pred_pd = pd.DataFrame()
pred_pd['Id'] = id_test
pred_pd['SalePrice'] = np.exp((pred_rf+pred_sgd)/2.000000)

pred_pd.head(5)
#pred_pd.to_csv('submission_mrig_v3.csv',index=False)


# In[ ]:





# **For some more advanced studies inluding Feature Engineering and methods like Stacking, Boosting and Voting have a look at my second kernel on this competition which I will link here soon**

# In[ ]:





# In[ ]:





# In[ ]:




