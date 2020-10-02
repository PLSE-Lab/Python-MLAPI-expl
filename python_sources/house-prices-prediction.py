#!/usr/bin/env python
# coding: utf-8

# # EDSA CPT Group 12 Linear Regression
# ## kaggle House Prices: Advanced Regression Techniques

# ----
# **Team Members**: Simphiwe Dangazela | Lakhiwe Liwani | Zipho Matiso | Purity Molala | Niel Smith

# ----
# # Introduction

# The goals of this notebook is to analyse the data provided in order to accurately fit applicable machine learning models that can be used to predict the price of a house based on certain features.

# ----
# # Load libraries and modules

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

from scipy import stats

from scipy.special import boxcox1p

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

plt.style.use('seaborn-paper')


# ----
# # Load the data sets

# The first step is to load the data sets provided by the competition into dataframes. This includes a training set that will be used to train and validate the models. A test set is provided that must be used to make predictions to be submitted for a score.

# In[ ]:


# # Run this cell for local machine.
# df_train = pd.read_csv('train.csv', index_col='Id')
# df_test = pd.read_csv('test.csv', index_col='Id')
# sample_sub = pd.read_csv('sample_submission.csv', index_col='Id')


# In[ ]:


# Run this cell for kaggle kernel.
import os
print(os.listdir("../input"))

df_train = pd.read_csv('../input/train.csv', index_col='Id')
df_test = pd.read_csv('../input/test.csv', index_col='Id')
sample_sub = pd.read_csv('../input/sample_submission.csv', index_col='Id')


# In[ ]:


# Do not truncate columns and rows of displayed dataframes.
pd.options.display.max_columns = None
# pd.options.display.max_rows = None


# In[ ]:


# The saleprice of every house is to be predicted and assign as the target.
target = df_train[['SalePrice']]


# In[ ]:


# Test and train data sets are joined along the index.
data = pd.concat([df_train, df_test], sort=False)


# ----
# # EDA
# The Exploratory Data Analysis helps to understand what data cleaning and transformation steps are required. This helps to decided on what machine learning models can be considered.

# ----
# ## The data

# In[ ]:


# Find the number of rows and columns.
pd.DataFrame([[df_train.shape[0], df_train.shape[1]],
              [df_test.shape[0], df_test.shape[1]] ],
             index=['Train Set', 'Test Set'],
             columns=['Number of rows', 'Number of columns'])


# In[ ]:


# List all the columns (features).
data.columns


# In[ ]:


# Find columns that are only in training set and not in the test set (submission set).
[col for col in df_train.columns if col not in df_test.columns]


# In[ ]:


data.info()


# In[ ]:


data.head()


# In[ ]:


# List the columns with numerical types.
numerical = data.dtypes[data.dtypes != object].index.tolist()
pd.DataFrame(numerical, index=range(1, len(numerical)+1), columns=['Numerical'])


# In[ ]:


# List the columns with non-numerical types.
categorical = data.dtypes[data.dtypes == object].index.tolist()
pd.DataFrame(categorical, index=range(1, len(categorical)+1), columns=['Categorical'])


# There are a number of columns that are missing values. The columns are also of different datatypes (object, int, float) that will have to be inspected to be used as numerical or categorical features. Some of the columns might need to be changed from numerical to categorical and vice versa.

# In[ ]:


data.describe()


# ----
# ## The target: Sale Price

# The house feature that must be predicted is its house price and will therefore be referred to as the target variable.
# An inspection of the sale prices is done.
# 
# Only the test set contains sale price observations as the sale price must be predicted for the test set.

# In[ ]:


# Plot the SalePrice observations in ascending order.
plt.figure(figsize=(10,5))
plt.scatter(range(len(target)), target.sort_values(by='SalePrice'))
plt.xlabel('Observation',fontsize=20)
plt.ylabel('SalePrice',fontsize=20)
plt.title('SalePrice Observations in Ascending Order',fontsize=20)

plt.rc('xtick',labelsize=15)
plt.rc('ytick',labelsize=15)
plt.tight_layout()
# plt.savefig("saleprice_ordered.jpeg", dpi=150)


# The training set has entries with very high house prices that will cause a skew distrubtion of data.

# In[ ]:


# Plot the distribution of the SalePrice observations and compare to a normal distribution.
sns.set(rc={'figure.figsize':(10,5)})
ax = sns.distplot(target, fit=stats.norm, kde=False, hist_kws={"label": "Target"}, fit_kws={"label": "Normal Fit"})
ax.axes.set_title("SalePrice Distribution Plot",fontsize=30)
ax.set_xlabel("SalePrice/Target Variable",fontsize=20)
ax.tick_params(labelsize=15)
ax.legend(fontsize='x-large', title_fontsize='40')
plt.tight_layout()
# plt.savefig("sales_dist_skew.jpeg", dpi=150)


# The SalePrice values are not normally distributed, but rather skewed to the right/positively skewed (Skewness value > 0).
# This means that the very expensive houses are seen as outliers. The following boxplot and descriptive statistics confirms the skewness.

# In[ ]:


# Calculate the skewness.
stats.skew(target)


# In[ ]:


# Plot a boxplot for Saleprice.
sns.set(rc={'figure.figsize':(10,5)})
ax = sns.boxplot(x = target)
ax.axes.set_title("Box-and-Whisker",fontsize=30)
ax.set_xlabel("SalePrice/Target Variable",fontsize=20)
ax.tick_params(labelsize=15)
plt.tight_layout()
# plt.savefig("skew_box.jpeg", dpi=150)


# In[ ]:


target.describe()


# The sale price data will be log transformed.

# In[ ]:


target_log1p = np.log1p(target)


# In[ ]:


# Plot the log transformed distribution of the SalePrice observations and compare to a normal distribution.
sns.set(rc={'figure.figsize':(10,5)})
ax = sns.distplot(target_log1p, fit=stats.norm, kde=False, hist_kws={"label": "Log of Target"}, fit_kws={"label": "Normal Fit"})
ax.axes.set_title("Transformed SalePrice Distribution Plot",fontsize=30)
ax.set_xlabel("SalePrice/Target Variable",fontsize=20)
ax.legend()
ax.tick_params(labelsize=15)
plt.tight_layout()
# plt.savefig("saleprice_dist_correct.jpeg", dpi=150)


# In[ ]:


# Plot the transformed saleprice boxplot.
sns.set(rc={'figure.figsize':(10,5)})
ax = sns.boxplot(x = target_log1p)
ax.set(xlabel='SalePrice/Target Variable',title='Transformed Box-and-Whisker')
plt.tight_layout()
# plt.savefig("norm_box.jpeg", dpi=150)


# In[ ]:


stats.skew(target_log1p)


# ----
# ## Correlations

# A correlation matrix is created and plotted as a heatmap to see which variables are correlated to the target and to each other.

# In[ ]:


# Create a correlation matrix between all numerical features of the training data.
correlation_matrix = df_train.corr().round(2)
sns.set(rc={'figure.figsize':(15,12)})
ax = sns.heatmap(data=correlation_matrix, vmin=-1, vmax=1, cmap='RdBu')
ax.tick_params(labelsize=20)
plt.tight_layout()
# plt.savefig("heatmap.jpeg", dpi=150)


# At this stage only numerical features can be seen. When the categorical features are transformed into numerical features the correlations can be checked again.

# The correlation heatmap shows that OverallQual and GrLivArea are two of the strongest predictors for the sale price. Multicollinearity  between predictors can also be observed. YearBuilt and GarageYBlt are strongly correlated as can be expected, since a house and its garage are built around the same year.

# In[ ]:


strongest = abs(correlation_matrix['SalePrice']).sort_values(ascending=False)[:11].index
strong_matrix = correlation_matrix.loc[strongest,strongest]
strong_matrix


# If the features strongest correlated to SalePrice is plotted, the variation in the relationships can be seen.

# In[ ]:


# Plot scatters of strongest numerical features vs SalePrice.
plt.figure(figsize=(25, 15))
strong_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF']

for i, col in enumerate(strong_features):
    plt.subplot(2, len(strong_features)/2 , i+1)
    sns.regplot(df_train[col], df_train['SalePrice'], line_kws={"color": "red"})
    plt.title(col, fontsize=25)
    plt.xlabel(col, fontsize=25)
    plt.ylabel('SalePrice', fontsize=25)
plt.tight_layout()
# plt.savefig("strong_correlations.jpeg", dpi=150)


# All three features have nearly linear relationships with the target variable - SalePrice.
# The residual variation in OverallQual looks to be constant where in general living area it fans out at higher prices and areas.

# In[ ]:


df_train.columns


# In[ ]:


# Plot linear regression fit scatter plots for first 20 numerical.
plt.figure(figsize=(40,40))
i = 1
for col in numerical[:20]:
    plt.subplot(4,5,i)
    sns.regplot(df_train[col], df_train['SalePrice'], line_kws={"color": "red"})
    plt.title(col, fontsize=25)
    plt.xlabel(col, fontsize=25)
    plt.ylabel('SalePrice', fontsize=25)
    plt.xticks(fontsize=14)
    plt.xticks(fontsize=14)
    i = i + 1


# In[ ]:


# Plot linear regression fit scatter plots for the rest of numerical.
plt.figure(figsize=(40,40))
i = 1
for col in numerical[20:]:
    plt.subplot(4,5,i)
    sns.regplot(df_train[col], df_train['SalePrice'], line_kws={"color": "red"})
    plt.title(col, fontsize=25)
    plt.xlabel(col, fontsize=25)
    plt.ylabel('SalePrice', fontsize=25)
    i = i + 1


# In[ ]:


# Plot swarmplots of first 2 categorical features.
# It is too slow to plot all.
plt.figure(figsize=(10,15))
i = 1
for col in categorical[:2]:
    plt.subplot(2,1,i)
    sns.swarmplot(df_train[col], df_train['SalePrice'])
    plt.title(col, fontsize=25)
    plt.xlabel(col, fontsize=25)
    plt.ylabel('SalePrice', fontsize=25)
    i = i + 1


# ----
# # Filling in missing values

# In[ ]:


# Create a dataframe showing missing values for train, test and combined.
missing_values = pd.concat([df_train.isna().sum(), df_test.isna().sum()], axis=1, keys=['Train', 'Test'], sort=False)
missing_values['Combined Data'] = missing_values.sum(axis=1)
missing_values['Percentage Missing'] = missing_values['Combined Data']/data.shape[0]*100
missing_values = missing_values[missing_values['Combined Data']>0].sort_values(by='Combined Data', ascending=False)
missing_values


# In[ ]:


# Plot the missing quantities.
ax = missing_values[['Percentage Missing']].plot.bar(figsize=(10,8), width=1)
plt.ylabel('Percentage Missing', fontsize=25)
plt.xlabel('Feature', fontsize=25)
plt.title('Missing Values', fontsize=25)
ax.tick_params(labelsize=15)
plt.tight_layout()
# plt.savefig("missing data.jpeg", dpi=150)


# In[ ]:


# # Open, read and close the desciption text file.
# # Only use this cell locally.
# f = open('data_description.txt', 'r')
# for line in f:
#     print(line)
# f.close()


# The PoolQC has the most missing values. The different values for PoolQC is inspected and also the related feaure, namely, PoolArea. The data description text file is also consulted.

# In[ ]:


data['PoolQC'].value_counts()


# In[ ]:


data['PoolArea'].value_counts()


# In[ ]:


# Check where there is a pool but no QC value.
pools_noQC = data[(data['PoolQC'].isna()) & (data['PoolArea']!=0)][['PoolQC']].index
data.loc[pools_noQC, ['PoolQC','PoolArea','OverallQual', 'OverallCond', 'ExterQual', 'ExterCond']]


# The missing values for PoolQC without an actual pool can be set to 'none' while the three unknown pool qualities with PoolAreas will be assumed to be average/typical.

# In[ ]:


data.loc[pools_noQC, 'PoolQC'] = 'TA'


# In[ ]:


data['PoolQC'].fillna('none', inplace=True)


# MiscFeature, Alley, and Fence are replaced with 'none' when there is no instance of the feature as per the description text.

# In[ ]:


data.loc[data[(data['MiscFeature'].isna()) & (data['MiscVal']!=0)].index][['MiscFeature','MiscVal']]


# In[ ]:


data['MiscFeature'].value_counts()


# In[ ]:


data.loc[2550, 'MiscFeature'] = 'Othr'


# In[ ]:


data['MiscFeature'].fillna('none', inplace=True)


# In[ ]:


data['Alley'].fillna('none', inplace=True)


# In[ ]:


data['Fence'].fillna('none', inplace=True)


# The missing fireplace qualities is checked for values>0.

# In[ ]:


data[(data['FireplaceQu'].isna()) & (data['Fireplaces']!=0)]


# In[ ]:


data[(data['FireplaceQu'].isna()) & (data['Fireplaces']==0)].index


# In[ ]:


data['FireplaceQu'].fillna('none', inplace=True)


# Investigating how the house prices vary per neighborhood, and looking at the diferences between the mean and median per neighborhood helps to indicate what value to use to fill the NaN for LotFrontage.

# In[ ]:


pd.concat([data.groupby('Neighborhood').median()['LotFrontage'],
           data.groupby('Neighborhood').mean()['LotFrontage'],
           data.groupby('Neighborhood').mean()['SalePrice']],axis=1)


# In[ ]:


data.groupby('Neighborhood')['LotFrontage'].describe()


# The LotFrontage is taken as the median value.

# In[ ]:


data['LotFrontage'].mean(), data['LotFrontage'].median()


# In[ ]:


data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))


# For missing basement values it is first checked if there is a basement or not.

# In[ ]:


# Inspect basement related values.
data[data['BsmtFinSF1'].isna()][['BsmtQual',
                                 'BsmtCond',
                                 'BsmtExposure',
                                 'BsmtFinType1',
                                 'BsmtFinType2',
                                 'BsmtFinSF1',
                                 'BsmtFinSF2',
                                 'BsmtUnfSF',
                                 'TotalBsmtSF',
                                 'BsmtHalfBath',
                                 'BsmtFullBath']]


# By inspecting the different values in the basement features it can be seen where 'No' or 'none' is applicable.

# In[ ]:


data['BsmtQual'].value_counts()


# In[ ]:


data['BsmtCond'].value_counts()


# In[ ]:


data['BsmtExposure'].value_counts()


# In[ ]:


data['BsmtFinType1'].value_counts()


# In[ ]:


data['BsmtFinType2'].value_counts()


# The string 'none' will be used when the feautre is not there. In the case of BsmtExposure the term 'No' is used to give meaning when there is a basement but no exposure.

# In[ ]:


data.loc[2121, ['BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']] = 'none'
data.loc[2121, ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtHalfBath','BsmtFullBath']] = 0


# In[ ]:


data[data['BsmtHalfBath'].isna()][['BsmtQual',
                                   'BsmtCond',
                                   'BsmtExposure',
                                   'BsmtFinType1',
                                   'BsmtFinType2',
                                   'BsmtFinSF1',
                                   'BsmtFinSF2',
                                   'BsmtUnfSF',
                                   'TotalBsmtSF',
                                   'BsmtHalfBath',
                                   'BsmtFullBath']]


# In[ ]:


data.loc[2189, ['BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']] = 'none'
data.loc[2189, ['BsmtHalfBath','BsmtFullBath']] = 0


# In[ ]:


data[( data['BsmtFinType2'].isna() ) & ( ~( data['BsmtFinType1'].isna() ) )]    [['BsmtQual',
      'BsmtCond',
      'BsmtExposure',
      'BsmtFinType1',
      'BsmtFinType2',
      'BsmtFinSF1',
      'BsmtFinSF2',
      'BsmtUnfSF',
      'TotalBsmtSF',
      'BsmtHalfBath',
      'BsmtFullBath']]


# In this case there is a basement so a mode value is more applicable.

# In[ ]:


data['BsmtFinType2'].value_counts()


# In[ ]:


data.loc[333, ['BsmtFinType2']] = 'Unf'


# In[ ]:


data[( data['BsmtFinType2'].isna() ) & ( ( data['BsmtFinType1'].isna() ) )][['BsmtQual',
                                   'BsmtCond',
                                   'BsmtExposure',
                                   'BsmtFinType1',
                                   'BsmtFinType2',
                                   'BsmtFinSF1',
                                   'BsmtFinSF2',
                                   'BsmtUnfSF',
                                   'TotalBsmtSF',
                                   'BsmtHalfBath',
                                   'BsmtFullBath']].describe()


# In[ ]:


# Find missing exposure values with actual basements.
miss_no_BSMTexp = data[( ~( data['BsmtFinType2'].isna() ) & ~( data['BsmtFinType1'].isna() ) ) &                       ( data['BsmtExposure'].isna() )]                       [['BsmtQual',
                       'BsmtCond',
                       'BsmtExposure',
                       'BsmtFinType1',
                       'BsmtFinType2',
                       'BsmtFinSF1',
                       'BsmtFinSF2',
                       'BsmtUnfSF',
                       'TotalBsmtSF',
                       'BsmtHalfBath',
                       'BsmtFullBath']].index


# In[ ]:


miss_no_BSMTexp


# In[ ]:


data.loc[miss_no_BSMTexp, ['BsmtExposure']] = 'No'


# In[ ]:


miss_no_BSMTqual = data[( ~( data['BsmtFinType2'].isna() ) & ~( data['BsmtFinType1'].isna() ) ) &                       ( data['BsmtQual'].isna() )]                       [['BsmtQual',
                       'BsmtCond',
                       'BsmtExposure',
                       'BsmtFinType1',
                       'BsmtFinType2',
                       'BsmtFinSF1',
                       'BsmtFinSF2',
                       'BsmtUnfSF',
                       'TotalBsmtSF',
                       'BsmtHalfBath',
                       'BsmtFullBath']].index


# In[ ]:


data.loc[miss_no_BSMTqual][['BsmtQual',
                       'BsmtCond',
                       'BsmtExposure',
                       'BsmtFinType1',
                       'BsmtFinType2',
                       'BsmtFinSF1',
                       'BsmtFinSF2',
                       'BsmtUnfSF',
                       'TotalBsmtSF',
                       'BsmtHalfBath',
                       'BsmtFullBath']]


# In[ ]:


data.loc[[2218,2219], ['BsmtQual']] = 'TA'


# In[ ]:


data[ ~( data['BsmtFinType2'].isna() ) & ( data['BsmtCond'].isna() )]                       [['BsmtQual',
                       'BsmtCond',
                       'BsmtExposure',
                       'BsmtFinType1',
                       'BsmtFinType2',
                       'BsmtFinSF1',
                       'BsmtFinSF2',
                       'BsmtUnfSF',
                       'TotalBsmtSF',
                       'BsmtHalfBath',
                       'BsmtFullBath']]


# In[ ]:


data.loc[[2041,2186,2525], ['BsmtCond']] = 'TA'


# In[ ]:


data[data['BsmtFinType2'].isna()]                       [['BsmtQual',
                       'BsmtCond',
                       'BsmtExposure',
                       'BsmtFinType1',
                       'BsmtFinType2',
                       'BsmtFinSF1',
                       'BsmtFinSF2',
                       'BsmtUnfSF',
                       'TotalBsmtSF',
                       'BsmtHalfBath',
                       'BsmtFullBath']].describe()


# The rest of the missing basement values there is no basement.

# In[ ]:


data['BsmtQual'].fillna(value='none', inplace=True)
data['BsmtCond'].fillna(value='none', inplace=True)
data['BsmtExposure'].fillna(value='none', inplace=True)
data['BsmtFinType1'].fillna(value='none', inplace=True)
data['BsmtFinType2'].fillna(value='none', inplace=True)


# In[ ]:


data['Electrical'].fillna(value=data['Electrical'].mode()[0], inplace=True)


# In[ ]:


data['Exterior1st'].value_counts()


# In[ ]:


data['Exterior1st'].fillna(value='Other', inplace=True)
data['Exterior2nd'].fillna(value='Other', inplace=True)


# In[ ]:


data.loc[1550:1560, 'KitchenQual']


# In[ ]:


data['KitchenQual'].fillna(method='ffill', inplace=True)


# In[ ]:


data[data['SaleType'].isna()]


# In[ ]:


data['SaleType'].value_counts()


# In[ ]:


data['SaleType'].fillna(value='Oth', inplace=True)


# In[ ]:


data[data['GarageCars'].isna()][['GarageCars',
                                 'GarageArea',
                                 'GarageType',
                                 'GarageYrBlt',
                                 'GarageFinish',
                                 'GarageQual',
                                 'GarageCond']]


# In[ ]:


garage_info = data[data['GarageType']=='Detchd'][['GarageArea', 'GarageCars']].describe()


# In[ ]:


data['GarageArea'].fillna(value=garage_info.loc['mean','GarageArea'], inplace=True)
data['GarageCars'].fillna(value=int(garage_info.loc['mean','GarageCars']), inplace=True)


# In[ ]:


data[data['Neighborhood'] == 'IDOTRR']['Utilities'].unique()


# In[ ]:


data[data['Utilities'].isna()]


# In[ ]:


data.loc[1916, 'Utilities'] = 'AllPub'
data.loc[1946, 'Utilities'] = 'AllPub'


# In[ ]:


data['MasVnrType'].fillna(value='none', inplace=True)
data['MasVnrArea'].fillna(value=0, inplace=True)


# In[ ]:


data['Functional'].value_counts()


# In[ ]:


data['Functional'].fillna(value='Typ', inplace=True)


# In[ ]:


data['MSZoning'].value_counts()


# In[ ]:


data[data['MSZoning'].isna()][['Neighborhood','MSSubClass']]


# In[ ]:


data[data['Neighborhood']=='IDOTRR'].groupby(['MSSubClass','MSZoning'])['MSZoning'].count()


# In[ ]:


data.loc[2217, 'MSZoning'] = 'C (all)'
data.loc[1916, 'MSZoning'] = 'RM'
data.loc[2251, 'MSZoning'] = 'RM'


# In[ ]:


data[data['Neighborhood']=='Mitchel'].groupby(['MSSubClass','MSZoning'])['MSZoning'].count()


# In[ ]:


data.loc[2905, 'MSZoning'] = 'RL'


# The garage relate missing values are also fisrt checked if a garage is present. For the rest of the no garage entries 'none' and 0 are used.

# In[ ]:


data[~( data['GarageType'].isna() ) & ( data['GarageYrBlt'].isna() )][['GarageCars',
  'GarageArea',
  'GarageType',
  'GarageYrBlt',
  'GarageFinish',
  'GarageQual',
  'GarageCond',
  'YearBuilt',
  'YearRemodAdd',
  'MiscFeature',
  'MSZoning']]


# In[ ]:


data.loc[2127,'GarageYrBlt'] = 1910
data.loc[2577,'GarageYrBlt'] = 1923


# In[ ]:


data.loc[2127,'GarageCond'] = data['GarageCond'].mode()[0]
data.loc[2577,'GarageCond'] = data['GarageCond'].mode()[0]


# In[ ]:


data.loc[2127,'GarageFinish'] = data['GarageFinish'].mode()[0]
data.loc[2577,'GarageFinish'] = data['GarageFinish'].mode()[0]


# In[ ]:


data.loc[2127,'GarageQual'] = data['GarageQual'].mode()[0]
data.loc[2577,'GarageQual'] = data['GarageQual'].mode()[0]


# In[ ]:


data[~( data['GarageCars'].isna() ) & ( data['GarageType'].isna() )]                                                                     [['GarageCars',
                                                                       'GarageArea',
                                                                       'GarageType',
                                                                       'GarageYrBlt',
                                                                       'GarageFinish',
                                                                       'GarageQual',
                                                                       'GarageCond',
                                                                       'SalePrice']].shape


# In[ ]:


data['GarageType'].fillna(value='none', inplace=True)
data['GarageFinish'].fillna(value='none', inplace=True)
data['GarageQual'].fillna(value='none', inplace=True)
data['GarageCond'].fillna(value='none', inplace=True)


# In[ ]:


GarageYrBlt_na_index = data[data['GarageYrBlt'].isna()].index
data.loc[GarageYrBlt_na_index, 'GarageYrBlt'] = data.loc[GarageYrBlt_na_index, 'YearBuilt']


# In[ ]:


data[['GarageYrBlt']].sort_values(by='GarageYrBlt').head()


# In[ ]:


data[['GarageYrBlt']].sort_values(by='GarageYrBlt').tail()


# This value for GarageYrBlt is most likely a typing error and supposed to be 2007 as the year remodeling was done.

# In[ ]:


data.loc[[2593]]


# In[ ]:


data.loc[2593,'GarageYrBlt'] = data.loc[2593,'YearRemodAdd']


# In[ ]:


data.isna().sum().sum()


# In[ ]:


data.drop('SalePrice', axis=1, inplace=True)


# In[ ]:


data.isna().sum().sum()


# ----
# # Categorical data handling

# In[ ]:


data.shape


# In[ ]:


data.head()


# The sublclass, month sold and year sold are picked up as numerical values but carry categorical meaning and is therefore converted to strings.

# In[ ]:


data = data.replace( {'MSSubClass' : {num: 'MSC'+ str(num) for num in list(data['MSSubClass'].unique())}} )


# In[ ]:


data = data.replace( {'MoSold' : {num: 'M'+ str(num) for num in list(data['MoSold'].unique())}} )


# In[ ]:


data = data.replace( {'YrSold' : {num: 'YS'+ str(num) for num in list(data['YrSold'].unique())}} )


# In[ ]:


numerical = data.dtypes[data.dtypes != object].index.tolist()
categorical = data.dtypes[data.dtypes == object].index.tolist()


# The following simplifications are done to avoid redundant columns.
# The simplifications gave slightly better results.

# In[ ]:


# Combine all house areas and bath quantities.
data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
data['Baths'] = data['BsmtFullBath'] + 0.5*data['BsmtHalfBath'] + data['FullBath'] + 0.5*data['HalfBath']

data.drop(['TotalBsmtSF',
           '1stFlrSF',
           '2ndFlrSF',
           'BsmtFullBath',
           'BsmtHalfBath',
           'FullBath',
           'HalfBath'], axis=1, inplace=True)

# Combine basement areas.
data['BsmtTotSF'] = data['BsmtFinSF1'] + data['BsmtFinSF2'] + data['BsmtUnfSF'] + data['LowQualFinSF']
data.drop(['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'LowQualFinSF'], axis=1, inplace=True)

# Combine porch/deck type areas.
data['TotalPorch'] = data['WoodDeckSF'] + data['OpenPorchSF'] + data['EnclosedPorch'] + data['3SsnPorch'] + data['ScreenPorch']
data.drop(['WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch'], axis=1, inplace=True)

# For the following features with scarce scatter plots, a binary yes or no is assigned to.
data['Pool'] = data['PoolArea'].apply(lambda x: 'Y' if x > 0 else 'N')
data.drop('PoolArea', axis=1, inplace=True)

data['Garage'] = data['GarageArea'].apply(lambda x: 'Y' if x > 0 else 'N')
data.drop('GarageArea', axis=1, inplace=True)

data['FireP'] = data['Fireplaces'].apply(lambda x: 'Y' if x > 0 else 'N')
data.drop('Fireplaces', axis=1, inplace=True)

data['kitchen'] = data['KitchenAbvGr'].apply(lambda x: 'Y' if x > 0 else 'N')
data.drop('KitchenAbvGr', axis=1, inplace=True)

data['MiscValHigh'] = data['MiscVal'].apply(lambda x: 'Y' if x > 100 else 'N')
data.drop('MiscVal', axis=1, inplace=True)


# In[ ]:


numerical = data.dtypes[data.dtypes != object].index.tolist()
categorical = data.dtypes[data.dtypes == object].index.tolist()


# # Transform Skew Numerical Features

# The numerical features with skewed data is log transformed.

# In[ ]:


def skew_features(df):
    skew_list = []
    for col in df.columns:
        if ( (stats.skew(df[col])) > 0.6 ):
            skew_before = stats.skew(df[col])
            skew_B = stats.skew(boxcox1p(df[col], stats.boxcox_normmax(df[col] + 1)))
            if (skew_B < skew_before) and (skew_B > 0):
                skew_list.append(col)
                
            print('BoxCox', col, skew_before, skew_B)
    return skew_list


# In[ ]:


skew_list = skew_features(data[numerical])


# In[ ]:


skew_list


# In[ ]:


# Plot skew features before transforming.
i = 1
sns.set(rc={'figure.figsize':(10,30)})
for col in skew_list:
    plt.subplot(len(skew_list),1,i)
    sns.distplot(data[col], fit=stats.norm, kde=False, hist_kws={"label": col}, fit_kws={"label": "Normal Fit"})
    i = i + 1
plt.tight_layout()
# plt.savefig("skew_before.jpeg", dpi=150)


# In[ ]:


for col in skew_list:
    data[col] = boxcox1p(data[col], stats.boxcox_normmax(data[col] + 1))


# In[ ]:


# Plot transformed features.
i = 1
sns.set(rc={'figure.figsize':(10,30)})
for col in skew_list:
    plt.subplot(len(skew_list),1,i)
    sns.distplot(data[col], fit=stats.norm, kde=False, hist_kws={"label": col}, fit_kws={"label": "Normal Fit"})
    i = i + 1
plt.tight_layout()
# plt.savefig("skew_after.jpeg", dpi=150)


# ----
# # Outliers

# Only the very extreme values are used to drop row entries. If more rows are dropped the test scores of the models decrease significantly.

# In[ ]:


def find_outliers(df):
    listd = []
    for col in df.columns:
        upper = df[col].mean() + 5*(df[col].std())
        lower = df[col].mean() - 5*(df[col].std())
        
        dropp = df[(df[col] > upper) | (df[col] < lower)][[col]].index.tolist()
        listd.append(dropp)
        print(col, dropp)
        #df.drop(dropp, axis=0, inplace=True)
    return list(set([item for sublist in listd for item in sublist]))


# In[ ]:


to_drop = find_outliers(data.loc[:1460, numerical])


# In[ ]:


len(to_drop)/df_test.shape[0]*100


# In[ ]:


data.drop(to_drop, inplace=True)
target_log1p.drop(to_drop, inplace=True)


# In[ ]:


data.loc[:1460].shape


# In[ ]:


target_log1p.shape


# ----
# # Encode Categorical Features

# In[ ]:


def encode_data(input_df, cols):
    
    input_df = pd.get_dummies(input_df, columns=cols, drop_first=True, prefix=cols, prefix_sep='_')
    
    return input_df


# In[ ]:


encoded_df = encode_data(data, data[categorical].columns.tolist())


# # Datasets for different models

# Two datasets are made: one for linear regression correlation filtering and one for the rest.

# In[ ]:


LData = encoded_df.loc[:1460]
RData = encoded_df[:]


# # Multiple Linear Regression with Variable Selection

# ## Correlated Columns

# In[ ]:


LData['SalePrice'] = target_log1p['SalePrice'].values


# In[ ]:


linear_cols = LData.corr()[abs(LData.corr()['SalePrice']) > 0.5][['SalePrice']].sort_values(by='SalePrice', ascending=False).index.tolist()
LData.loc[:, linear_cols].corr() > 0.8


# In[ ]:


multi_coll = ['GrLivArea','GarageYrBlt','SalePrice']


# In[ ]:


use_cols = [col for col in linear_cols if col not in multi_coll]
use_cols.append('SalePrice')
LData.loc[:, use_cols].corr()['SalePrice']


# In[ ]:


use_cols.remove('SalePrice')


# In[ ]:


use_cols


# In[ ]:


RData.isna().sum().sum()


# ## Evaluation Functions

# The function below is used to plot the performance of a model.

# In[ ]:


# Function called after a models has been trained and predicted values.
# This plots visuals to assess the model's performance.
def evaluate_models(model_label):
    plt.figure(figsize=(15,10))
    plt.subplot(2,3,1)
    plt.scatter(range(len(y_train)), y_train.sort_values(by='SalePrice'), label='Actual Train SalePrice Ordered')
    plt.scatter(range(len(y_pred_train)), sorted(y_pred_train), label='Predicted Train SalePrice Ordered')
    plt.legend()
    plt.xlabel('Observation')
    plt.ylabel('SalePrice')
    
    plt.subplot(2,3,2)
    errors = np.array(y_pred_train.reshape(-1,1) - y_train)
    errors = np.round(errors, 2)
    plt.hist(errors, label='Train Errors')
    plt.legend()
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    
    plt.subplot(2,3,3)
    plt.scatter(y_train, y_pred_train.reshape(-1,1), label='y_train vs y_train_predicted')
    plt.legend()
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    
    plt.subplot(2,3,4)
    plt.scatter(range(len(y_test)), y_test.sort_values(by='SalePrice'), label='Actual Test SalePrice Ordered')
    plt.scatter(range(len(y_pred_test)), sorted(y_pred_test), label='Predicted Test SalePrice Ordered')
    plt.legend()
    plt.xlabel('Observation')
    plt.ylabel('SalePrice')
    
    plt.subplot(2,3,5)
    errors = np.array(y_pred_test.reshape(-1,1) - y_test)
    errors = np.round(errors, 2)
    plt.hist(errors, label='Test Errors')
    plt.legend()
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    
    plt.subplot(2,3,6)
    plt.scatter(y_test, y_pred_test.reshape(-1,1), label='y_test vs y_test_predicted')
    plt.legend()
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    
    plt.tight_layout()
#     plt.savefig(model_label+".jpeg", dpi=150)
    plt.show()


# In[ ]:


# Function to calculate the MSE, R2 and LMSE metrics for y_actual and y_predicted.
def get_score(y_actual, y_predicted, label=''):
    MSE       = metrics.mean_squared_error(y_actual, y_predicted)
    R_squared = metrics.r2_score(y_actual, y_predicted)
    LMSE      = np.sqrt(MSE)
    
    print('-------'+label+'--------')
    print('MSE_'+label+':', MSE)
    print('R_squared_'+label+':', R_squared)
    print('LMSE_'+label+':', LMSE)


# ----
# ## Choose X, X_sub and y for Linear Model

# In[ ]:


y = target_log1p
X = RData.loc[:1460,use_cols]
X_sub = RData.loc[1461:,use_cols]


# ## Linear Model

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)


# In[ ]:


lm1 = LinearRegression()
lm1.fit(X_train, y_train)

y_pred_train = lm1.predict(X_train)
get_score(y_train, y_pred_train, 'Train')

y_pred_test = lm1.predict(X_test)
get_score(y_test, y_pred_test, 'Test')

evaluate_models('Linear')


# ----
# # Choose X, X_sub and y for Remaining Models

# In[ ]:


y = target_log1p
X = RData.loc[:1460]
X_sub = encoded_df.loc[1461:]


# # Decision Tree and Random Forest

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)


# In[ ]:


DTreg = DecisionTreeRegressor()
RFreg = RandomForestRegressor(n_estimators=100)

DTreg.fit(X_train, y_train)
RFreg.fit(X_train, y_train.values.ravel())

models = [DTreg, RFreg]
modelstr = ['Decision', 'RandomForest']

i = 0
for modl in models:
    print(modl)
    y_pred_train = modl.predict(X_train)
    get_score(y_train, y_pred_train, 'Train')
    
    y_pred_test = modl.predict(X_test)
    get_score(y_test, y_pred_test, 'Test')
    
    evaluate_models(modelstr[i])
    i = i + 1

    print()


# ----
# # Regularized Models

# The data is scaled to ensure the regularization penalties are applied fairly across all features and is not affected by a features range. The z standardization ensures a large enough range.

# In[ ]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_sub_scaled = scaler.fit_transform(X_sub)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)


# 
# ----
# ## Modelling

# In[ ]:


ridge = Ridge(alpha=10)
lasso = Lasso(alpha=0.003)

ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)

models = [ridge, lasso ]
modelstr = ['ridge', 'lasso']

i = 0
for modl in models:
    print(modl)
    y_pred_train = modl.predict(X_train)
    get_score(y_train, y_pred_train, 'Train')
    
    y_pred_test = modl.predict(X_test)
    get_score(y_test, y_pred_test, 'Test')
    
    evaluate_models(modelstr[i])
    i = i + 1

    print()


# In[ ]:


coeff = pd.DataFrame(lasso.coef_, X.columns, columns=['Coefficient'])


# In[ ]:


coeff.sort_values(by='Coefficient', ascending=False)[:10]


# In[ ]:


coeff.sort_values(by='Coefficient', ascending=True)[:10]


# In[ ]:


lasso.intercept_


# ----
# # Submission

# In[ ]:


# Use the sample submission file for the correct format and index.
sub = sample_sub


# In[ ]:


# Replace the data with the lasso model (best performance) predictions
sub['SalePrice'] = np.exp(lasso.predict(X_sub_scaled))-1


# In[ ]:


# save the submission
sub.to_csv('submission.csv')

