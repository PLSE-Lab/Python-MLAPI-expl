#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


train = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')
test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')
sample_submission = pd.read_csv('/kaggle/input/home-data-for-ml-course/sample_submission.csv')


# In[ ]:


train.shape, test.shape


# In[ ]:


train = train.drop('Id', 1)
test = test.drop('Id', 1)


# ## Define Util Functions

# In[ ]:


def compare_box(col):
    print(col)
    print("Train MAX : {0},  Test MAX : {1}".format(train[col].max(), test[col].max()))
    print("Train MIN : {0},  Test MIN : {1}".format(train[col].min(), test[col].min()))
    print("Train MEAN: {0:.2f},  Test MEAN: {1:.2f}".format(train[col].mean(), test[col].mean()))
    print("Train STD : {0:.2f},  Test STD : {1:.2f}".format(train[col].std(), test[col].std()))
    print("Train NaN : {0},  Test STD : {1}".format(train[col].isnull().sum(), test[col].isnull().sum()))
    print("----"*10)
    fg, ax = plt.subplots(figsize=(12, 6))
    fg.add_subplot(1, 2, 1)
    sns.boxplot(y=train[col])
    plt.xlabel('Train')
    fg.add_subplot(1, 2, 2)
    sns.boxplot(y=test[col])
    plt.xlabel('Test')


# ## Explore Data

# In[ ]:


# Here I want to select numerical data columns
num_col = train.select_dtypes(exclude='object').drop('SalePrice', 1).columns


# In[ ]:


vis_col = len(num_col)/4+1


# In[ ]:


fg, ax = plt.subplots(figsize=(12, 18))
for i, col in enumerate(num_col):
    fg.add_subplot(vis_col, 4, i+1)
    sns.distplot(train[col].dropna())
    plt.xlabel(col)
    
plt.tight_layout()
plt.show()


# In[ ]:


fg, ax = plt.subplots(figsize=(12, 18))
for i, col in enumerate(num_col):
    fg.add_subplot(vis_col, 4, i+1)
    sns.scatterplot(x=train[col], y=train['SalePrice'])
    plt.xlabel(col)
    
plt.tight_layout()
plt.show()


# ## Explore NaN Values of Numerical Data

# In[ ]:


train[num_col].isnull().sum().sort_values(ascending=False).head()


# In[ ]:


test[num_col].isnull().sum().sort_values(ascending=False).head(15)


# In[ ]:


train['MasVnrArea'].fillna(train['MasVnrArea'].mode()[0], inplace=True)


# In[ ]:


test['TotalBsmtSF'].fillna(0, inplace=True)
test['MasVnrArea'].fillna(test['MasVnrArea'].mode()[0], inplace=True)
test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].mode()[0], inplace=True)
test['BsmtFullBath'].fillna(test['BsmtFullBath'].mode()[0], inplace=True)
test['GarageArea'].fillna(test['GarageArea'].mode()[0], inplace=True)
test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mode()[0], inplace=True)
test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].mode()[0], inplace=True)
test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].mode()[0], inplace=True)
test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mode()[0], inplace=True)
test['GarageCars'].fillna(test['GarageCars'].mode()[0], inplace=True)


# In[ ]:


train['LotFrontage'].fillna(train.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.median()), inplace=True)
test['LotFrontage'].fillna(test.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.median()), inplace=True)


# In[ ]:


train.corr()['GarageYrBlt'].sort_values(ascending=False).head()


# GarageYrBlt & YearBuilt Correlation is over 0.82
# 
# So, I just want to drop the columne 'GarageYrBlt'

# In[ ]:


train['GarageYrBlt'].fillna(0, inplace=True)
test['GarageYrBlt'].fillna(0, inplace=True)


# In[ ]:


train[num_col].isnull().sum().sort_values(ascending=False).head()


# In[ ]:


test[num_col].isnull().sum().sort_values(ascending=False).head()


# ## Outliers

# In[ ]:


fg, ax = plt.subplots(figsize=(12, 18))
for i, col in enumerate(num_col):
    fg.add_subplot(9, 4, i+1)
    sns.boxplot(y=train[col])
    plt.xlabel(col)
    
plt.tight_layout()
plt.show()


# **1. SalePrice**

# In[ ]:


plt.figure(figsize=(7, 6))
sns.boxplot(y='SalePrice', data=train)


# In[ ]:


train.loc[train['SalePrice'] > 700000, :]


# In[ ]:


train = train.loc[train['SalePrice'] < 700000, :]


# **2. LotArea**

# In[ ]:


compare_box('LotArea')


# In[ ]:


train = train.loc[train['LotArea'] < 100000, :]


# In[ ]:


compare_box('LotArea')


# **3. TotalBsmtSF**

# In[ ]:


compare_box('TotalBsmtSF')


# Previously I removed outlier over 5000 at this column.
# 
# But it results out to be bad performace of predicting outliers of test set.
# Evaluation is RMSE
# 

# **4. LotFrontage**

# In[ ]:


compare_box('LotFrontage')


# In[ ]:


train = train.loc[train['LotFrontage'] < 210, :]


# In[ ]:


compare_box('LotFrontage')


# **5. MasVnrArea**

# In[ ]:


compare_box('MasVnrArea')


# In[ ]:


train = train.loc[train['MasVnrArea'] < 1300, :]


# In[ ]:


compare_box('MasVnrArea')


# **6. BsmtFinSF1**

# In[ ]:


compare_box('BsmtFinSF1')


# In[ ]:


train = train.loc[train['BsmtFinSF1'] < 2000, :]


# In[ ]:


compare_box('BsmtFinSF1')


# **7. BsmtFinSF2**

# In[ ]:


compare_box('BsmtFinSF2')


# In[ ]:


train = train.loc[train['BsmtFinSF2'] < 1200, :]


# In[ ]:


compare_box('BsmtFinSF2')


# In[ ]:


train.shape, test.shape


# ## Make change to categorical column

# In[ ]:


train['MoSold'].value_counts()


# In[ ]:


train['MoSold'] = train['MoSold'].astype('object')
test['MoSold'] = test['MoSold'].astype('object')


# In[ ]:


train['YrSold'].value_counts()


# In[ ]:


train['YrSold'] = train['YrSold'].astype('object')
test['YrSold'] = test['YrSold'].astype('object')


# In[ ]:


num_col = train.select_dtypes(exclude='object').drop('SalePrice', 1).columns
num_col


# In[ ]:


len(num_col)


# ## Explore Categorical Data

# In[ ]:


cat_col = train.select_dtypes(include='object').columns
cat_col


# In[ ]:


len(cat_col), len(num_col)


# In[ ]:


vis_col = len(cat_col)/4 +1
fg, ax = plt.subplots(figsize=(12, 18))

for i, col in enumerate(cat_col):
    fg.add_subplot(vis_col, 4, i+1)
    sns.countplot(train[col])
    plt.xlabel(col)

plt.tight_layout()
plt.show()


# ## Explore NaN Values

# In[ ]:


train[cat_col].isnull().sum().sort_values(ascending=False).head(20)


# In[ ]:


test[cat_col].isnull().sum().sort_values(ascending=False).head(20)


# In[ ]:


train['RoofMatl'].value_counts()


# In[ ]:


test['RoofMatl'].value_counts()


# In[ ]:


train['RoofMatl_clean'] = train['RoofMatl'].apply(lambda x: x if x == 'CompShg' else 'Other')
test['RoofMatl_clean'] = test['RoofMatl'].apply(lambda x: x if x == 'CompShg' else 'Other')


# In[ ]:


train['Alley'].value_counts()


# In[ ]:


test['Alley'].value_counts()


# In[ ]:


train['Alley'].fillna('None', inplace=True)
test['Alley'].fillna('None', inplace=True)


# In[ ]:


train['Alley_bool'] = train['Alley'].apply(lambda x: 0 if x == 'None' else 1)
test['Alley_bool'] = test['Alley'].apply(lambda x: 0 if x == 'None' else 1)


# In[ ]:


train['Alley_bool'].value_counts()


# In[ ]:


test['Alley_bool'].value_counts()


# In[ ]:


train['Electrical'].value_counts()


# In[ ]:


test['Electrical'].value_counts()


# In[ ]:


train['Electrical'].fillna(train['Electrical'].mode()[0], inplace=True)


# In[ ]:


train['Electrical_clean'] = train['Electrical'].apply(lambda x: x if x == 'SBrkr' else 'Fuse')
test['Electrical_clean'] = test['Electrical'].apply(lambda x: x if x == 'SBrkr' else 'Fuse')


# In[ ]:


train['Electrical_clean'].value_counts()


# In[ ]:


test['Electrical_clean'].value_counts()


# In[ ]:


train['MasVnrType'].value_counts()


# In[ ]:


test['MasVnrType'].value_counts()


# In[ ]:


train['MasVnrType'].isnull().sum()


# In[ ]:


test['MasVnrType'].isnull().sum()


# In[ ]:


train['MasVnrType'].fillna(train['MasVnrType'].mode()[0], inplace=True)
test['MasVnrType'].fillna(test['MasVnrType'].mode()[0], inplace=True)


# In[ ]:


test['MSZoning'].isnull().sum()


# In[ ]:


test['MSZoning'].value_counts()


# In[ ]:


test['MSZoning'].fillna(test['MSZoning'].mode()[0], inplace=True)


# In[ ]:


train['Functional'].value_counts()


# In[ ]:


test['Functional'].value_counts()


# In[ ]:


train['Functional'].isnull().sum()


# In[ ]:


test['Functional'].isnull().sum()


# In[ ]:


test['Functional'].fillna(test['Functional'].mode()[0], inplace=True)


# In[ ]:


train['Functional_clean'] = train['Functional'].apply(lambda x: x if x =='Typ' else 'Other')
test['Functional_clean'] = test['Functional'].apply(lambda x: x if x =='Typ' else 'Other')


# In[ ]:


train['Utilities'].value_counts()


# In[ ]:


test['Utilities'].value_counts()


# Here, train.Utilies has only one value of 'NoSeWa'
# and if I make that row drop, then Utilites column will have only one unified value.
# 
# So, I just want to drop the column

# In[ ]:


test['Utilities'].fillna(test['Utilities'].mode()[0], inplace=True)


# In[ ]:


train['Exterior2nd'].isnull().sum()


# In[ ]:


test['Exterior2nd'].isnull().sum()


# In[ ]:


train['Exterior2nd'].value_counts()


# In[ ]:


test['Exterior2nd'].value_counts()


# Here, I can see that train dataset's data composition is quite diffenrent than test dataset's

# In[ ]:


ext_other = [
    'Stone',
    'AsphShn',
    'Other',
    'CBlock',
    'ImStucc',
    'Brk Cmn'
]


# In[ ]:


train['Exterior2nd'] = train['Exterior2nd'].apply(lambda x: 'Other' if x in ext_other else x)
test['Exterior2nd'] = test['Exterior2nd'].apply(lambda x: 'Other' if x in ext_other else x)


# In[ ]:


train['Exterior2nd'].fillna('Other', inplace=True)
test['Exterior2nd'].fillna('Other', inplace=True)


# In[ ]:


train['Exterior2nd'].value_counts()


# In[ ]:


test['Exterior2nd'].value_counts()


# In[ ]:


train['SaleType'].value_counts()


# In[ ]:


test['SaleType'].value_counts()


# For the SaleType,
# 
# there are many value categories with less than 10 counts.
# 
# So, I just want to simplify it

# In[ ]:


saletype_other = [
    'ConLD',
    'ConLw',
    'ConLI',
    'CWD',
    'Oth',
    'Con'
]


# In[ ]:


train['SaleType'] = train['SaleType'].apply(lambda x: x if x not in saletype_other else 'Other')
test['SaleType'] = test['SaleType'].apply(lambda x: x if x not in saletype_other else 'Other')


# In[ ]:


test['SaleType'].fillna('Other', inplace=True)


# In[ ]:


train['SaleType'].isnull().sum()


# In[ ]:


test['SaleType'].isnull().sum()


# In[ ]:


train['SaleType'].value_counts()


# In[ ]:


test['SaleType'].value_counts()


# In[ ]:


train['KitchenQual'].value_counts()


# In[ ]:


test['KitchenQual'].value_counts()


# In[ ]:


train['KitchenQual'].isnull().sum()


# In[ ]:


test['KitchenQual'].isnull().sum()


# In[ ]:


test['KitchenQual'].fillna('TA', inplace=True)


# In[ ]:


train['Exterior1st'].value_counts()


# In[ ]:


test['Exterior1st'].value_counts()


# In[ ]:


ext_other = [
    'Stone',
    'BrkComm',
    'ImStucc',
    'AsphShn',
    'Other',
    'CBlock',
]


# In[ ]:


train['Exterior1st'] = train['Exterior1st'].apply(lambda x: 'Other' if x in ext_other else x)
test['Exterior1st'] = test['Exterior1st'].apply(lambda x: 'Other' if x in ext_other else x)


# In[ ]:


train['Exterior1st'].fillna('Other', inplace=True)
test['Exterior1st'].fillna('Other', inplace=True)


# In[ ]:


train['Exterior1st'].isnull().sum()


# In[ ]:


test['Exterior1st'].isnull().sum()


# In[ ]:


train['MiscFeature'].value_counts()


# In[ ]:


train['MiscFeature'].fillna('None', inplace=True)
test['MiscFeature'].fillna('None', inplace=True)


# In[ ]:


train['MiscFeature_bool'] = train['MiscFeature'].apply(lambda x: 1 if x == 'None' else 0)
test['MiscFeature_bool'] = test['MiscFeature'].apply(lambda x: 1 if x == 'None' else 0)


# In[ ]:


train['Alley'].value_counts()


# In[ ]:


train['Alley'].fillna('None', inplace=True)
test['Alley'].fillna('None', inplace=True)


# In[ ]:


train['Fence'].value_counts()


# In[ ]:


test['Fence'].value_counts()


# In[ ]:


train['Fence'].fillna('None', inplace=True)
test['Fence'].fillna('None', inplace=True)


# In[ ]:


train['FireplaceQu'].value_counts()


# In[ ]:


test['FireplaceQu'].value_counts()


# In[ ]:


train['FireplaceQu'].fillna('None', inplace=True)
test['FireplaceQu'].fillna('None', inplace=True)


# In[ ]:


train[cat_col].isnull().sum().sort_values(ascending=False).head(20)


# In[ ]:


test[cat_col].isnull().sum().sort_values(ascending=False).head(20)


# In[ ]:


train['GarageQual'].value_counts()


# In[ ]:


train['GarageCond'].value_counts()


# In[ ]:


train['GarageQual'].fillna('None', inplace=True)
test['GarageQual'].fillna('None', inplace=True)


# In[ ]:


train['GarageQual_TA'] = train['GarageQual'].apply(lambda x: x if x == 'TA' else 'Other')
test['GarageQual_TA'] = test['GarageQual'].apply(lambda x: x if x == 'TA' else 'Other')


# In[ ]:


train['GarageFinish'].value_counts()


# In[ ]:


test['GarageFinish'].value_counts()


# In[ ]:


train['GarageFinish'].fillna('None', inplace=True)
test['GarageFinish'].fillna('None', inplace=True)


# In[ ]:


train['GarageType'].value_counts()


# In[ ]:


test['GarageType'].value_counts()


# In[ ]:


train['GarageType'].fillna('None', inplace=True)
test['GarageType'].fillna('None', inplace=True)


# In[ ]:


train['BsmtFinType2'].value_counts()


# In[ ]:


train['PoolQC'].fillna('None', inplace=True)
test['PoolQC'].fillna('None', inplace=True)


# In[ ]:


train['GarageCond'].fillna('None', inplace=True)
test['GarageCond'].fillna('None', inplace=True)


# In[ ]:


train['BsmtExposure'].fillna('None', inplace=True)
test['BsmtExposure'].fillna('None', inplace=True)


# In[ ]:


train['BsmtCond'].fillna('None', inplace=True)
test['BsmtCond'].fillna('None', inplace=True)


# In[ ]:


train['BsmtQual'].fillna('None', inplace=True)
test['BsmtQual'].fillna('None', inplace=True)


# In[ ]:


train['BsmtFinType1'].fillna('None', inplace=True)
test['BsmtFinType1'].fillna('None', inplace=True)


# In[ ]:


train['BsmtFinType2'].fillna('None', inplace=True)
test['BsmtFinType2'].fillna('None', inplace=True)


# In[ ]:


train[cat_col].isnull().sum().sort_values(ascending=False).head()


# In[ ]:


test[cat_col].isnull().sum().sort_values(ascending=False).head()


# ## Add Boolean columns

# In[ ]:


train['WoodDeckSF_bool'] = train['WoodDeckSF'].apply(lambda x: 1 if x > 0 else 0)
test['WoodDeckSF_bool'] = test['WoodDeckSF'].apply(lambda x: 1 if x > 0 else 0)


# In[ ]:


train['OpenPorchSF_bool'] = train['OpenPorchSF'].apply(lambda x: 1 if x > 0 else 0)
test['OpenPorchSF_bool'] = test['OpenPorchSF'].apply(lambda x: 1 if x > 0 else 0)


# In[ ]:


train['EnclosedPorch_bool'] = train['EnclosedPorch'].apply(lambda x: 1 if x > 0 else 0)
test['EnclosedPorch_bool'] = test['EnclosedPorch'].apply(lambda x: 1 if x > 0 else 0)


# In[ ]:


train['3SsnPorch_bool'] = train['3SsnPorch'].apply(lambda x: 1 if x > 0 else 0)
test['3SsnPorch_bool'] = test['3SsnPorch'].apply(lambda x: 1 if x > 0 else 0)


# In[ ]:


train['ScreenPorch_bool'] = train['ScreenPorch'].apply(lambda x: 1 if x > 0 else 0)
test['ScreenPorch_bool'] = test['ScreenPorch'].apply(lambda x: 1 if x > 0 else 0)


# In[ ]:


train['PoolArea_bool'] = train['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
test['PoolArea_bool'] = test['PoolArea'].apply(lambda x: 1 if x > 0 else 0)


# In[ ]:


train['FirePlaces_bool'] = train['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
test['FirePlaces_bool'] = test['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)


# ## Further Tuning

# In[ ]:


def draw_rel(df):
    plt.figure(figsize=(9, 6))
    sns.regplot(y=train['SalePrice'], x=df)


# In[ ]:


area_map = {
    0: 'GrLivArea',
    1: 'TotalBsmtSF',
    2: 'LotArea',
    3: 'GarageArea',
}


# In[ ]:


for col in area_map.values():
    draw_rel(train[col])


# make LotArea -> LotArea_log
# 
# I expect it to be more linear reg

# In[ ]:


train['LotArea_log'] = np.log1p(train['LotArea'])
test['LotArea_log'] = np.log1p(test['LotArea'])


# In[ ]:


draw_rel(train['LotArea_log'])


# In[ ]:


area_map = {
    0: 'GrLivArea',
    1: 'TotalBsmtSF',
    2: 'LotArea_log',
    3: 'GarageArea',
}


# In[ ]:


to_draw = {}
to_add_col = {}


# In[ ]:


for i in range(4):
    for j in range(i+1, 4):
        col_name = area_map[i]+"_"+area_map[j]+"_sum"
        col_value = train[area_map[i]] +train[area_map[j]]
        corr_val = col_value.corr(train['SalePrice'])
        print("CORR: {0} ===> {1}".format(col_name, corr_val))
        to_add_col[col_name] = col_value
        to_draw[col_name] = col_value


# In[ ]:


for i in range(4):
    for j in range(i+1, 4):
        col_name = area_map[i]+"_"+area_map[j]+"_mul"
        col_value = train[area_map[i]] * train[area_map[j]]
        corr_val = col_value.corr(train['SalePrice'])
        print("CORR: {0} ===> {1}".format(col_name, corr_val))
        to_add_col[col_name] = col_value
        to_draw[col_name] = col_value


# In[ ]:


for i in range(4):
    for j in range(i+1, 4):
        for z in range(j+1, 4):
            col_name = area_map[i]+"_"+area_map[j]+"_"+area_map[z]+"_sum"
            col_value = train[area_map[i]]+train[area_map[j]]+train[area_map[z]]
            corr_val = col_value.corr(train['SalePrice'])
            print("CORR: {0} ===> {1}".format(col_name, corr_val))
            to_add_col[col_name] = col_value
            to_draw[col_name] = col_value


# In[ ]:


for i in range(4):
    for j in range(i+1, 4):
        for z in range(j+1, 4):
            col_name = area_map[i]+"_"+area_map[j]+"_"+area_map[z]+"_mul"
            col_value = np.sqrt(train[area_map[i]]*train[area_map[j]]*train[area_map[z]])
            corr_val = col_value.corr(train['SalePrice'])
            print("CORR: {0} ===> {1}".format(col_name, corr_val))
            to_add_col[col_name] = col_value
            to_draw[col_name] = col_value


# In[ ]:


fg, ax = plt.subplots(figsize=(18,18))
# fg, ax = plt.subplots()
n_col = 4
n_row = len(to_draw) // n_col
cnt = 1
for col, value in to_draw.items():
    fg.add_subplot(n_row, n_col, cnt)
    sns.regplot(x=value, y=train['SalePrice'])
    plt.xlabel(col)
    cnt+=1
plt.tight_layout()
plt.show()


# In[ ]:


key_df = pd.DataFrame(to_add_col)
key_df


# In[ ]:


plt.figure(figsize=(12, 12))
sns.heatmap(key_df.corr())
plt.tight_layout()
plt.show()


# Adding too many produced columns will result in overfitting problem.
# 
# So, I just want to add feature which is highly correlated to 'SalePrice'

# In[ ]:


train['GrLivArea_TotalBsmtSF_GarageArea_sum'] = train['GrLivArea'] + train['TotalBsmtSF'] +train['GarageArea']
train['LotArea_log_GarageArea_mul'] = train['LotArea_log'] * train['GarageArea']


# In[ ]:


test['GrLivArea_TotalBsmtSF_GarageArea_sum'] = test['GrLivArea'] + test['TotalBsmtSF'] +test['GarageArea']
test['LotArea_log_GarageArea_mul'] = test['LotArea_log'] * test['GarageArea']


# In[ ]:


train.shape, test.shape


# ### Other Area

# In[ ]:


other_area_map = {
    0: 'WoodDeckSF',
    1: 'OpenPorchSF',
    2: 'EnclosedPorch',
    3: '3SsnPorch',
    4: 'ScreenPorch',
    5: 'PoolArea',
}


# In[ ]:


to_draw = {
    'WoodDeckSF': train['WoodDeckSF'],
    'OpenPorchSF': train['OpenPorchSF'],
    'EnclosedPorch': train['EnclosedPorch'],
    '3SsnPorch': train['3SsnPorch'],
    'ScreenPorch': train['ScreenPorch'],
    'PoolArea': train['PoolArea'],
}


# In[ ]:


for i in range(6):
    for j in range(i+1, 6):
        col_name = other_area_map[i]+"_"+other_area_map[j]+"_sum"
        col_value = train[other_area_map[i]] +train[other_area_map[j]]
        corr_val = col_value.corr(train['SalePrice'])
        print("CORR: {0} ===> {1}".format(col_name, corr_val))
        if corr_val > 0.3:
            to_add_col[col_name] = col_value
        to_draw[col_name] = col_value


# In[ ]:


for i in range(6):
    for j in range(i+1, 6):
        for z in range(j+1, 6):
            col_name = other_area_map[i]+"_"+other_area_map[j]+"_" + other_area_map[z] +"_sum"
            col_value = train[other_area_map[i]] +train[other_area_map[j]] + train[other_area_map[z]]
            corr_val = col_value.corr(train['SalePrice'])
            print("CORR: {0} ===> {1}".format(col_name, corr_val))
            if corr_val > 0.3:
                to_add_col[col_name] = col_value
            to_draw[col_name] = col_value


# In[ ]:


for i in range(6):
    for j in range(i+1, 6):
        col_name = other_area_map[i]+"_"+other_area_map[j]+"_mul"
        col_value = train[other_area_map[i]]*train[other_area_map[j]]
        corr_val = col_value.corr(train['SalePrice'])
        print("CORR: {0} ===> {1}".format(col_name, corr_val))
        if corr_val > 0.3:
            to_add_col[col_name] = col_value
        to_draw[col_name] = col_value


# In[ ]:


for i in range(6):
    for j in range(i+1, 6):
        for z in range(j+1, 6):
            col_name = other_area_map[i]+"_"+other_area_map[j]+"_" + other_area_map[z] +"_mul"
            col_value = train[other_area_map[i]] * train[other_area_map[j]] * train[other_area_map[z]]
            corr_val = col_value.corr(train['SalePrice'])
            print("CORR: {0} ===> {1}".format(col_name, corr_val))
            if corr_val > 0.3:
                to_add_col[col_name] = col_value
            to_draw[col_name] = col_value


# In[ ]:


fg, ax = plt.subplots(figsize=(18,12))
# fg, ax = plt.subplots()
n_col = 6
n_row = len(to_add_col) // n_col + 1
cnt = 1
for col, value in to_add_col.items():
    fg.add_subplot(n_row, n_col, cnt)
    sns.regplot(x=value, y=train['SalePrice'])
    plt.xlabel(col)
    cnt+=1
plt.tight_layout()
plt.show()


# In[ ]:


key_df = pd.DataFrame(to_add_col)
key_df


# In[ ]:


plt.figure(figsize=(12, 12))
sns.heatmap(key_df.corr())
plt.tight_layout()
plt.show()


# ## Bathroom Count

# In[ ]:


train['FullBathCount'] = train['BsmtFullBath'] + train['FullBath']
train['HalfBathCount'] = train['HalfBath'] + train['BsmtHalfBath']


# In[ ]:


test['FullBathCount'] = test['BsmtFullBath'] + test['FullBath']
test['HalfBathCount'] = test['HalfBath'] + test['BsmtHalfBath']


# ## Overall Rating

# In[ ]:


train['OverallMean'] = (train['OverallCond']+ train['OverallQual'])/2
test['OverallMean'] = (test['OverallCond']+ test['OverallQual'])/2


# ## Mixture w/ OverallQual and OverallCond

# In[ ]:


train['GrLivArea_OverallQual_mul'] = train['GrLivArea_TotalBsmtSF_GarageArea_sum'] * train['OverallQual'].astype(int)
test['GrLivArea_OverallQual_mul'] = test['GrLivArea_TotalBsmtSF_GarageArea_sum'] * test['OverallQual'].astype(int)


# In[ ]:


train['GrLivArea_OverallCond_mul'] = train['GrLivArea_TotalBsmtSF_GarageArea_sum'] * train['OverallCond'].astype(int)
test['GrLivArea_OverallCond_mul'] = test['GrLivArea_TotalBsmtSF_GarageArea_sum'] * test['OverallCond'].astype(int)


# In[ ]:


compare_box('GrLivArea_OverallQual_mul')


# ## Check up Missing Values

# In[ ]:


train.isnull().sum().sort_values(ascending=False).head()


# In[ ]:


test.isnull().sum().sort_values(ascending=False).head()


# In[ ]:


## Columns to remove
to_remove_cols = [
    'GarageYrBlt',
    'Utilities',
    'Street',
    'SalePrice',
    'PoolQC',
    'Id',
]

