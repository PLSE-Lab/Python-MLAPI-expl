#!/usr/bin/env python
# coding: utf-8

# # House Prices: Advanced Regression Techniques

# **Competition Description**
# 
# ![](https://www.reno.gov/Home/ShowImage?id=7739&t=635620964226970000)
# Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.
# 
# With **79** explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.
# 
# Practice Skills  
# Creative feature engineering   
# Advanced regression techniques like random forest and gradient boosting  
# 
# **Acknowledgments**  
# The [Ames Housing dataset](http://www.amstat.org/publications/jse/v19n3/decock.pdf) was compiled by Dean De Cock for use in data science education. It's an incredible alternative for data scientists looking for a modernized and expanded version of the often cited Boston Housing dataset. 
# 
# You can download the txt file here: [**download**](https://www.kaggle.com/c/5407/download/data_description.txt)

# ## We'll be going throught following topics :
# * EDA with Pandas and Seaborn  
# <li>Find features with strong correlation to target  
# <li>Data Wrangling, convert categorical to numerical  
# <li>apply the basic Regression models of sklearn  
# <li>use gridsearchCV to find the best parameters for each model  
# <li>compare the performance of the Regressors and choose best one  

# **The notebook is organized as follows:**
# 
# * **[Part 0: Imports, Settings and switches, Global functions](#Part-0-:-Imports,-Settings,-Functions)**  
# import libraries  
# settings for number of cross validations  
# * **[define functions that are used often](#common used function)**  
# 
# * **[Part 1: Exploratory Data Analysis](#Part-1:-Exploratory-Data-Analysis)**  
# 1.1 Get an overview of the features (numerical and categorical) and first look on the target variable SalePrice  
# [shape, info, head and describe](#shape,-info,-head-and-describe)  
# [Distribution of the target variable SalePrice](#The-target-variable-:-Distribution-of-SalePrice)  
# [Numerical and Categorical features](#Numerical-and-Categorical-features)  
# [List of features with missing values](#List-of-features-with-missing-values) and Filling missing values  
# [log transform](#log-transform)  
# 1.2 Relation of all features to target SalePrice  
# [Seaborn regression plots for numerical features](#Plots-of-relation-to-target-for-all-numerical-features)  
# [List of numerical features and their correlation coefficient to target](#List-of-numerical-features-and-their-correlation-coefficient-to-target)  
# [Seaborn boxplots for categorical features](#Relation-to-SalePrice-for-all-categorical-features)  
# [List of categorical features and their unique values](#List-of-categorical-features-and-their-unique-values)  
# 1.3 Determine the columns that show strong correlation to target  
# [Correlation matrix 1](#Correlation-matrix-1) : all numerical features  
# Determine features with largest correlation to SalePrice_Log
# 
# 
# * **[Part 2: Data wrangling](#Part-2:-Data-wrangling)**  
# [Dropping all columns with weak correlation to SalePrice](#Dropping-all-columns-with-weak-correlation-to-SalePrice)  
# [Convert categorical columns to numerical](#Convert-categorical-columns-to-numerical)  
# [Checking correlation to SalePrice for the new numerical columns](#Checking-correlation-to-SalePrice-for-the-new-numerical-columns)  
# use only features with strong correlation to target  
# [Correlation Matrix 2 (including converted categorical columns)](#Correlation-Matrix-2-:-All-features-with-strong-correlation-to-SalePrice)  
# create datasets for ML algorithms  
# One Hot Encoder  
# [StandardScaler](#StandardScaler)
# 
# * **[Part 3: Scikit-learn basic regression models and comparison of results](#Part-3:-Scikit-learn-basic-regression-models-and-comparison-of-results)**  
# implement GridsearchCV with RMSE metric for Hyperparameter tuning  
# for these models from sklearn:  
# [Linear Regression](#Linear-Regression)  
# [Ridge](#Ridge)  
# [Lasso](#Lasso)  
# [Elastic Net](#Elastic-Net)  
# [Stochastic Gradient Descent](#SGDRegressor)  
# [DecisionTreeRegressor](#DecisionTreeRegressor)  
# [Random Forest Regressor](#RandomForestRegressor)  
# [KNN Regressor](#KNN-Regressor)  
# baed on RMSE metric, compare performance of the regressors with their optimized parameters,  
# then explore correlation of the predictions and make submission with mean of best models  
# Comparison plot: [RMSE of all models](#Comparison-plot:-RMSE-of-all-models)  
# [Correlation of model results](#Correlation-of-model-results)  
# Mean of best models
# 
# 
# Note on scores:  
# Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price. (Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Part-0-:-Imports,-Settings,-Functions

# In[ ]:


# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

#3ways to access system files
from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))
print(check_output(["ls", "/kaggle/input/house-prices-advanced-regression-techniques/"]).decode("utf8"))

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


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


# # common used function

# In[ ]:


def get_best_score(grid):
    """Function to return best score
    args : grid
    output : best_score"""
    
    
    best_score = np.sqrt(-grid.best_score_)
    print(best_score)    
    print(grid.best_params_)
    print(grid.best_estimator_)
    
    return best_score


# In[ ]:


def print_cols_large_corr(df,nr_c,targ):
    """
    Function to print columns with larger correlations
    args:
        df = dataframe
        nr_c = num of columns
        targ = target column
    """
    corr=df.corr()
    corr_abs=corr.abs()
    print(corr_abs.nlargest(nr_c, targ)[targ])


# In[ ]:


def plot_corr_matrix(df, nr_c, targ) :
    """
    Function to plot correlation matrix between variables and target
    args:
        df = dataframe
        nr_c = num of columns
        targ = target column
    """
    
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


# ![](http://)**Load data**

# In[ ]:


df_train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df_test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


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

# In[ ]:


print(df_train.shape)
df_train.info()


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


df_train.describe()


# In[ ]:


df_test.describe()


# In[ ]:


print(df_test.shape)
df_test.info()


# In[ ]:


print(df_train.shape)
print(df_test.shape)


# **obersvation: **
# In train set, we have 1460 rows and 79 dependent variables + 1 id column + 1 target variable
# In test set, we have 1459 rows and 78 dependent variables + 1 id column
# 
# df train has 81 columns (79 features + id and target SalePrice) and 1460 entries (number of rows or house sales)  
# df test has 80 columns (79 features + id) and 1459 entries  
# There is lots of info that is probably related to the SalePrice like the area, the neighborhood, the condition and quality.   
# Maybe other features are not so important for predicting the target, also there might be a strong correlation for some of the features (like GarageCars and GarageArea).
# For some columns many values are missing: only 7 values for Pool QC in df train and 3 in df test

# ### The target variable : Distribution of SalePrice

# In[ ]:


sns.distplot(df_train['SalePrice']);
#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())


# observations from the distribution of SalePrice:  
# <li>Deviate from the normal distribution.
# <li>Have appreciable positive skewness.
# <li>Show peakedness.
# 
# As we see, the target variable SalePrice is not normally distributed.  
# This can reduce the performance of the ML regression models because some assume normal distribution,   
# see [sklearn info on preprocessing](http://scikit-learn.org/stable/modules/preprocessing.html)  
# Therfore we make a log transformation, the resulting distribution looks much better.  

# **skewness and kurtosis** - These two statistics are called "shape" statistics, i.e., they describe the shape of the distribution.  
# 
# **Skewness(+/-)** is a measure of the symmetry in a distribution.  A symmetrical dataset will have a skewness equal to 0.  So, a normal distribution will have a skewness of 0.   Skewness essentially measures the relative size of the two tails.  
# 
# **Kurtosis** is a measure of the combined sizes of the two tails (heaviness of the tails).  It measures the amount of probability in the tails.  The value is often compared to the kurtosis of the normal distribution, which is equal to 3.  If the kurtosis is greater than 3, then the dataset has heavier tails than a normal distribution (more in the tails).  If the kurtosis is less than 3, then the dataset has lighter tails than a normal distribution (less in the tails).

# In[ ]:


# going from non-normal to logarithmic distribution
df_train['SalePrice_Log']=np.log(df_train['SalePrice'])
sns.distplot(df_train['SalePrice_Log']);
#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice_Log'].skew())
print("Kurtosis: %f" % df_train['SalePrice_Log'].kurt())
# dropping old SalePrice column
df_train.drop('SalePrice',axis=1,inplace=True)


# ### Numerical and Categorical features

# In[ ]:


numerical_feats = df_train.dtypes[df_train.dtypes != "object"].index
print("Number of Numerical features: ", len(numerical_feats))

categorical_feats = df_train.dtypes[df_train.dtypes == "object"].index
print("Number of Categorical features: ", len(categorical_feats))


# In[ ]:


print('Numerical columns in train set :')
print(df_train[numerical_feats].columns)
print("*"*100)
print('Categorical columns in train set: ')
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


# In[ ]:


# unique values in Pool
df_train.PoolQC.unique()


# **Filling missing values**  
# For a few columns there is lots of NaN entries.  
# However, reading the data description we find this is not missing data:  
# For PoolQC, NaN is not missing data but means no pool, likewise for Fence, FireplaceQu etc.  

# In[ ]:


# Columns which have NaN values present in them
nan_cols = [i for i in df_train.columns if df_train[i].isnull().any()]
print(len(nan_cols))
nan_cols


# In[ ]:


# columns where NaN values have meaning e.g. no pool etc.
cols_fillna = ['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu',
               'GarageQual','GarageCond','GarageFinish','GarageType', 'Electrical',
               'KitchenQual', 'SaleType', 'Functional', 'Exterior2nd', 'Exterior1st',
               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2',
               'MSZoning', 'Utilities']


# In[ ]:


len(cols_fillna)


# In[ ]:


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


# df_train['LotFrontage'].head()
cols=['LotFrontage','GarageYrBlt','MasVnrArea','SalePrice_Log','ExterCond']
for col in cols:
    print(df_train[i].dtype)


# In[ ]:


# fillna with mean for the remaining columns: LotFrontage, GarageYrBlt, MasVnrArea
df_train.fillna(df_train.mean(), inplace=True)
df_test.fillna(df_test.mean(), inplace=True)


# In[ ]:


total=df_train.isnull().sum().sort_values(ascending=False)
percent=(df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data=pd.concat([total,percent],axis=1,keys=['Total','Percent'])
missing_data.head()


# total = df_train.isnull().sum().sort_values(ascending=False)
# percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
# missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# missing_data.head(5)


# In[ ]:


#Now, we should have 0 Nan columns
# Columns which have NaN values present in them
nan_cols = [i for i in df_train.columns if df_train[i].isnull().any()]
print(len(nan_cols))
nan_cols


# ![](http://)![](http://)**Missing values in train data ?**

# In[ ]:


df_train.isnull().sum().sum()


# 1. 1. **Missing values in test data ?**

# In[ ]:


df_test.isnull().sum().sum()


# ### log transform
# Like the target variable, also some of the feature values are not normally distributed and it is therefore better to use log values in df_train and df_test. Checking for skewness and kurtosis:

# In[ ]:


numerical_feats


# In[ ]:


categorical_feats


# In[ ]:


len(numerical_feats)+len(categorical_feats)


# In[ ]:


for col in numerical_feats:
    print('{:15}'.format(col), 
          'Skewness: {:05.2f}'.format(df_train[col].skew()) , 
          '   ' ,
          'Kurtosis: {:06.2f}'.format(df_train[col].kurt())  
         )


# In[ ]:


# lets check skewness and kurtosis for GrLivArea
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


# transforming to make closer to normal dstribution (log) for GrLivArea and LotArea
for df in [df_train, df_test]:
    df['GrLivArea_Log'] = np.log(df['GrLivArea'])
    df.drop('GrLivArea', inplace= True, axis = 1)
    df['LotArea_Log'] = np.log(df['LotArea'])
    df.drop('LotArea', inplace= True, axis = 1)
    
    
    
numerical_feats = df_train.dtypes[df_train.dtypes != "object"].index


# In[ ]:


print(len(numerical_feats))
numerical_feats


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


sns.distplot(df_train['OverallQual'])


# In[ ]:


df_train = df_train.drop(
    df_train[(df_train['OverallQual']==10) & (df_train['SalePrice_Log']<12.3)].index)


# In[ ]:


df_train = df_train.drop(
    df_train[(df_train['GrLivArea_Log']>8.3) & (df_train['SalePrice_Log']<12.5)].index)


# **Find columns with strong correlation to target**  
# Only those with r > min_val_corr are used in the ML Regressors in Part 3  
# The value for min_val_corr can be chosen in global settings

# In[ ]:





# In[ ]:





# In[ ]:


# columns where NaN values have meaning e.g. no pool etc.
cols_fillna = ['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu',
               'GarageQual','GarageCond','GarageFinish','GarageType', 'Electrical',
               'KitchenQual', 'SaleType', 'Functional', 'Exterior2nd', 'Exterior1st',
               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2',
               'MSZoning', 'Utilities']


# In[ ]:


df_train.info()


# In[ ]:


df_train.isna()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


numerical_columns_train


# In[ ]:





# In[ ]:


df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


df_test= pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
df_train= pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df_submission=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')


# In[ ]:


df_test.head()


# In[ ]:


df=df_train
df.head()


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


# Analyse data
# clean data
# take significant columns
# make base model
# apply feature engineering
# try to beat model , use many model
# predict

