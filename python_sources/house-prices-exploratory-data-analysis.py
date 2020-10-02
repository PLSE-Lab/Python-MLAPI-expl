#!/usr/bin/env python
# coding: utf-8

# # House Prices Exploratory Data Analysis

# + Based on House Prices Data from [House Prices: Advanced Regression Techniques
# ](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
# + Data description available at [here](https://storage.googleapis.com/kaggle-competitions-data/kaggle/5407/data_description.txt?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1521905350&Signature=URtM6eNlWyfg7iGw1UQZ32S1wmFow5vIv7LRO9MVj%2FFT0tX4PpA3yAjF%2Bf8VoWBAwkZP5ag0pVO06V7bgAIp6rh0zjUK2qEhfbf76Pau6Ca7fS2BA75OrfHVcAyNQKySTyzwCk9Wcjk1rr9XqbMJIWqL08tWdcx%2Fro%2B72gfzo06GMcawmeWN58TZoH0ydpTpF817MkMh%2F%2FwXubZE1HP3OkX3HjeUVa9jOnyRZ%2FuIEfuelpKorIL%2BnZBxViw2BcuBd8K774glZFkB1mj3XeLXyshYrIB5IBPi2MtoZGdoE%2B5vouNsenASybCRBvFLBj4K5se%2Bi95dMYaxgmZJR2clQQ%3D%3D)

# ## Summary
# 
# - Dataset has 81 attributes and 1460 records in total.
# - Dataset has many type variables both of numerical (int64 or float64) and categorical (object) variables. These will need to handled with the transforming.
# - Data have a lot of NaN values. Consider to deal missing values. The missing data statistics:
# 
#   + Numerical Missing Values:
#   ```
#     LotFrontage 	 259 NaNs
#     GarageYrBlt 	 81 NaNs
#     MasVnrArea 	     8 NaNs
#   ```
#   + Categorical Missing Values:
#   ```
#     PoolQC 	         1453 NaNs
#     MiscFeature 	 1406 NaNs
#     Alley 	         1369 NaNs
#     Fence 	         1179 NaNs
#     FireplaceQu 	 690 NaNs
#     GarageCond 	     81 NaNs
#     GarageType 	     81 NaNs
#     GarageFinish 	 81 NaNs
#     GarageQual 	     81 NaNs
#     BsmtExposure 	 38 NaNs
#     BsmtFinType2 	 38 NaNs
#     BsmtFinType1 	 37 NaNs
#     BsmtCond 	     37 NaNs
#     BsmtQual 	     37 NaNs
#     MasVnrType 	     8 NaNs
#     Electrical 	     1 NaNs
#   ```
# - The target need to log1p to increase the performance.
# - There are a lot of features need to featuring. Will consider to deep dive and select the important features.
# - Some outliers need to remove to avoid overfitting. Will consider some attributes: `GrLivArea`, `GarageArea`, `TotalBsmtSF`, `1stFlrSF`

# ## Load libaries

# In[1]:


# data manipulation
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)

# plotting
import seaborn as sn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# setting params
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (30, 10),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}

sn.set_style('whitegrid')
sn.set_context('talk')

plt.rcParams.update(params)

# config for show max number of output lines
pd.options.display.max_colwidth = 600
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999

# pandas display data frames as tables
from IPython.display import display, HTML

# modeling utilities
import scipy.stats as stats
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict

import warnings
warnings.filterwarnings('ignore')


# ## Load dataset

# In[2]:


# Load train & test datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# ## Summarize Data

# In[3]:


# Display dimension of datasets
print('Train data shape', train.shape)
print('Test data shape', test.shape)


# Peek of the datasets

# In[4]:


train.head()


# In[5]:


test.head()


# In[6]:


# dataset summary stats
train.describe()


# ## Data Types

# In[7]:


# data types of attributes of train set
train.dtypes


# The Train dataset has:
# + 81 attributes in total and 1460 records
# + Dataset has many type variables both of numerical (int or float) and categorical variables. These will need to handled with care

# ## Missing values

# Need to deep dive into analysis NaN value of each column.

# In[8]:


# Count unique missing value of each column
for col in train.columns:
    if train[col].isnull().values.any():
        print(col)
        print(train[col].isnull().sum())


# In[9]:


total_missing = train.isnull().sum().sort_values(ascending=False)
ratio_missing = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total_missing, ratio_missing], axis=1, keys=['Total', 'Ratio'])
missing_data['Type'] = train[missing_data.index].dtypes

missing_data = missing_data[(missing_data['Total'] > 0)]

# display missing data
missing_data


# ### Numerical Missing values

# In[10]:


print('Numerical Missing Values:')
print('=========================')
[print(col_missing,  '\t', missing_data['Total'][col_missing], 'NaNs')  for col_missing in missing_data[(missing_data['Total'] > 0) &                                  (missing_data['Type'] != 'object')].index.values]
print('=========================')


# ### Categorical Missing values

# In[11]:


print('Categorical Missing Values:')
print('=========================')
[print(col_missing,  '\t', missing_data['Total'][col_missing], 'NaNs')  for col_missing in missing_data[(missing_data['Total'] > 0) &                                  (missing_data['Type'] == 'object')].index.values]
print('=========================')


# ## Analysis SalePrice that is the target of prediction

# In[12]:


train['SalePrice'].describe()


# In[13]:


train['SalePrice'].skew()


# In[14]:


train['SalePrice'].kurt()


# Visualizing the SalePrice distribution of train dataset

# In[15]:


plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
plt.hist(train['SalePrice'], color='blue')
plt.xlabel('SalePrice')
plt.show()


# Visualizing the SalePrice distribution of train dataset with log1p

# In[16]:


plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
plt.hist(np.log1p(train['SalePrice']), color='blue')
plt.xlabel('SalePrice')
plt.show()


# ## Numerical Features

# Matrix heatmap help to indentify correlation between the target and other features

# In[17]:


corr = train.select_dtypes(include=['float64', 'int64']).iloc[:,1:].corr()
f, ax = plt.subplots(figsize=(22, 22))
sn.heatmap(corr, vmax=.8, square=True)


# In[18]:


# Correlation between attributes with SalePrice
corr_list = corr['SalePrice'].sort_values(axis=0, ascending=False).iloc[1:]
corr_list


# In[19]:


# Scatter plotting the top related to SalePrice
plt.figure(figsize=(22, 22))
k = 6

for i in range(k):
    ii = '32'+str(i)
    plt.subplot(ii)
    feature = corr_list.index.values[i]
    plt.scatter(train[feature], train['SalePrice'], facecolors='none', edgecolors='k', s=75)
    sn.regplot(x=feature, y='SalePrice', data=train, scatter=False, color='b')
    ax=plt.gca()
    ax.set_ylim([0,800000])


# ## Visualize Correlated Attributes

# Visualizing top 10 related attributes to the target

# In[20]:


# Scatter plotting the variables most correlated with SalePrice
cols = corr.nlargest(10, 'SalePrice')['SalePrice'].index
sn.set()
sn.pairplot(train[cols], size=2.5)
plt.show()


# ## Categorical Features

# In[21]:


cat_df = train.select_dtypes(include=['object'])


# In[22]:


cat_df.shape


# In[23]:


cat_df.dtypes


# ### Box plotting for 15 first categorical attributes

# In[24]:


for cat in cat_df.dtypes[:15].index.values:
    plt.figure(figsize=(16, 22))
    plt.xticks(rotation=90)
    sn.boxplot(x=cat, y='SalePrice', data=train)    
    plt.show()


# Regarding the categorical features, we will consider following:
# 
# - Deal with missing values on categorical attributes
# - Transforming categorical attributes to numerical values (LabelEncode, LabelBinaryEncode, One-Hot-Encode, ...)
