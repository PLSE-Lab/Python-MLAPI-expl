#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder 
  
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
RANDOM_SEED = 101
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
train.head()


# In[ ]:


test_file = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
test_file.head()


# In[ ]:


submission = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")
submission.head()


# In[ ]:


train.columns


# In[ ]:


train.info()


# In[ ]:


train.describe()


# In[ ]:


target = 'SalePrice'
fig = plt.figure(figsize=(10,6))
sns.kdeplot(train['SalePrice'])


# In[ ]:


train['YearBuilt'].value_counts()


# In[ ]:


sns.distplot(train['YearBuilt'])
sns.set(font_scale=1)


# In[ ]:


missing_data = pd.DataFrame(train.isna().sum().sort_values(ascending=False))
missing_data.head(20)
missing_data[0]


# In[ ]:


train["Neighborhood"].value_counts()


# In[ ]:


plt.figure(figsize=(10,5))
train['MSZoning'].value_counts().plot.bar()


# In[ ]:


train['LandContour'].value_counts().plot.bar()


# In[ ]:


train['OverallQual'].value_counts().plot.bar()


# In[ ]:


sns.countplot(x='OverallQual',data=train)


# In[ ]:


sns.countplot(x='OverallCond',data=train)


# In[ ]:


sns.countplot(x="Foundation",data=train)


# In[ ]:


sns.kdeplot(train["TotalBsmtSF"],shade=True)


# In[ ]:


plt.figure(figsize=(15,6))
sns.scatterplot(x='TotalBsmtSF',y='SalePrice',data=train)


# In[ ]:


sns.scatterplot(x="GrLivArea",y="SalePrice",data=train)


# In[ ]:


sns.boxplot(x='OverallQual',y='SalePrice',data=train)


# In[ ]:


plt.figure(figsize=(10,8))
sns.scatterplot(x='YearBuilt',y="SalePrice",data=train)


# In[ ]:


plt.figure(figsize=(20,8))
sns.boxplot(x="Neighborhood",y="SalePrice",data=train)


# In[ ]:


sns.countplot(train["RoofStyle"])


# In[ ]:


sns.boxplot(x="RoofStyle",y='SalePrice',data=train)


# In[ ]:


sns.countplot(train['BsmtCond'])


# In[ ]:


sns.scatterplot(x="GarageArea",y="SalePrice",data=train)


# In[ ]:


sns.distplot(train['TotRmsAbvGrd'])


# In[ ]:


sns.boxplot(x="TotRmsAbvGrd",y="SalePrice",data=train)


# In[ ]:


train['TotRmsAbvGrd'].value_counts().plot.bar()


# In[ ]:


train['FullBath'].unique()


# In[ ]:


train['FullBath'].value_counts().plot.bar()


# In[ ]:


sns.boxplot(x="FullBath",y="SalePrice",data=train)


# In[ ]:


rel = pd.crosstab(train['TotRmsAbvGrd'],train['FullBath'])
print(rel)


# In[ ]:


train["HalfBath"].value_counts()


# In[ ]:


train["FullBath"].value_counts()


# In[ ]:


sns.kdeplot(train['GarageArea'])


# In[ ]:


train['GarageArea'].describe()


# In[ ]:


sns.scatterplot(x="GarageArea",y="SalePrice",data=train)


# In[ ]:


sns.countplot(train["GarageCars"])


# In[ ]:


sns.boxplot(x="GarageCars",y="SalePrice",data=train)


# In[ ]:


sns.boxplot(x="GarageCars",y="GarageArea",data=train)


# In[ ]:


sns.boxplot(x="MSSubClass",y="SalePrice",data=train)


# In[ ]:


train['MSSubClass'].value_counts().plot.bar()


# In[ ]:


sns.lineplot(x="YearRemodAdd",y="SalePrice",data=train)


# In[ ]:


train['HouseStyle'].value_counts().plot.bar()


# In[ ]:


sns.boxplot(x='HouseStyle',y='SalePrice',data=train)


# In[ ]:


cor = train.corr()['SalePrice']
cor.sort_values(ascending=False)


# important features
# 1. TotalBsmtSF(int)
# 2. GarageArea(int)
# 3. YearRemodAdd(int)
# 4. YearBuilt(int)
# 5. GrLivArea (int)
# 6. 1stFlrSF(int)
# 7. TotRmsAbvGrd(cetegory)
# 8. GarageCars(cetegory
# 9. OverallQual(cetegory)
# 10. FullBath(cetegory)
# 

# In[ ]:


null_values = train.isnull().sum()/train.shape[0]*100
null_values[(null_values>=30)]


# In[ ]:


null_values = train.isnull().sum()/train.shape[0]*100
null_values[(null_values>0) & (null_values<30)]


# In[ ]:


train.loc[:,['MasVnrType','Electrical']] = train[['MasVnrType','Electrical']].fillna(train[['MasVnrType','Electrical']].mode().iloc[0])
train['GarageYrBlt'] = train['GarageYrBlt'].fillna(train.YearBuilt)
train.GarageYrBlt.isna().sum()


# In[ ]:


train[['LotFrontage','MasVnrArea']] = train[['LotFrontage','MasVnrArea']].fillna(train[['LotFrontage','MasVnrArea']].mean().iloc[0])


# In[ ]:





# In[ ]:


impute_none = train[['Alley','FireplaceQu','PoolQC','Fence','MiscFeature','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','GarageFinish','GarageQual','GarageCond','GarageType']]
train.update(impute_none.fillna("None"))
train


# In[ ]:


train.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd'],axis=1,inplace=True)


# In[ ]:





# In[ ]:


num_cols = ["TotalBsmtSF","GarageArea","YearRemodAdd","YearBuilt","GrLivArea","1stFlrSF"]
cat_cols = ["TotRmsAbvGrd","GarageCars","OverallQual","FullBath",'HouseStyle']


# <h1>EXPLORATORY DATA ANALYSIS</h1>
# 
# <h3>UNIVARIATE ANALYSIS</h3>

# In[ ]:


j=1
fig = plt.figure(figsize=(25,6))
for col in num_cols:
    axs = fig.add_subplot(1,len(num_cols),j)
    axs = sns.distplot(train[col],label="Skewness: .%2f"%(train[col].skew()))
    axs.set_xlabel(col)
    axs.set_ylabel("frequency")
    axs.legend(loc='best')
    j+=1


# In[ ]:


fig = plt.figure(figsize=(50,15))
j = 1
for cat in cat_cols:
    ax = fig.add_subplot(1,len(cat_cols),j)
    ax = sns.countplot(x = cat, data=train)
    ax.set_xlabel(cat,fontsize=30)
    ax.set_ylabel("Frequency count",fontsize=30)
    plt.xticks(fontsize=30,rotation=90)
    plt.yticks(fontsize=30)
    j += 1  


# In[ ]:


fig = plt.figure(figsize=(10,5))
sns.pairplot(train[num_cols])


# In[ ]:


fig = plt.figure(figsize=(50,20))
j = 1
for col in num_cols:
    ax = fig.add_subplot(1,len(num_cols),j)
    ax = sns.scatterplot(train[col],train['SalePrice'])
    ax.set_xlabel(col,fontsize=30)
    ax.set_ylabel('SalePrice',fontsize=30)
    plt.xticks(fontsize=30,rotation=90)
    plt.yticks(fontsize=30,rotation=90)
    j += 1


# In[ ]:


for num_col in num_cols:
    fig = plt.figure(figsize = (30,10))
    j = 1
    for cat_col in cat_cols:
        ax = fig.add_subplot(1,len(cat_cols),j)
        sns.boxplot(y = train[num_col],
                    x = train[cat_col], 
                    data = train, 
                    ax = ax)
        ax.set_xlabel(cat_col,size=15)
        ax.set_ylabel(num_col,size=15)
        plt.xticks(fontsize=20,rotation=90)
        plt.yticks(fontsize=20,rotation=90)
        j = j + 1


# In[ ]:


j=1
fig = plt.figure(figsize=(30,6))
for cat in cat_cols:
    ax = fig.add_subplot(1,len(cat_cols),j)
    sns.boxplot(train[cat],train['SalePrice'],data=train)
    ax.set_xlabel(cat)
    ax.set_ylabel('Sale Price')
    j+=1


# In[ ]:


train.isna().mean().sort_values(ascending=False)
train_cp = train
p = pd.concat(objs=[train_cp.drop(columns='SalePrice'),test_file],axis=0)
p.isna().mean().sort_values(ascending=False)


# In[ ]:


l = []
for i in train.columns:
    if train[i].dtypes == np.object:
        l.append(i)
l


        


# In[ ]:


fig = plt.figure(figsize=(300,200))
for j in range(1,len(l)):
    ax = plt.subplot(16,3,j)
    sns.countplot(l[j],data=train)
    ax.set_xlabel(l[j],fontsize=50)
    ax.set_ylabel('Frequency',fontsize=50)
    plt.xticks(fontsize=50,rotation=90)
    plt.yticks(fontsize=50,rotation=90)
    


# In[ ]:


for i in l:
    print(train[i].value_counts().sort_values())


# ***Cetegorical features***
# 'MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCon','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition'
# 
# 
# 
# 
# 
# 

# In[ ]:



for i in l:
    df = pd.DataFrame()
    df[i + '_train']= train[i].value_counts()
    df[i + '_test'] = test_file[i].value_counts()
    print(df.head(15))


#    ****FEATURES TO BE DISCARDED****
#    
# 1. Alley
# 2. Utilities
# 3. Condition2
# 4. RoofMtl
# 5. Exterior2nd
# 6. PoolQC
# 7. MiscFeature
# 8. PoolArea
# 9. 2ndFlrSF
# 
#     ****FEATURE ENGINEERING ON THIS FEATURES CAN BE DONE!!****
#     
# 1. HouseStyle
# 2. RoofStyle
# 3. Exterior1st
# 4. Heating
# 5. Electrical
# 6. GarageQual
# 7. BedroomAbvGr
# 8. KitchenAbvGr 
# 9. TotRmsAbvGrd

# In[ ]:





# In[ ]:


l2 = []

for i in train.columns:
    if train.dtypes[i] == np.int64:
        l2.append(i)
l2
len(l2)


# In[ ]:


for i in l2:
    df = pd.DataFrame()
    if(len(list(train[i].unique()))<20):
        df[i + '_train']= train[i].value_counts()
        df[i + '_test'] = test_file[i].value_counts()
    print(df.head(20))


# MSSubClass,OverallQual,OverallCond,BsmtFullBath,BsmtHalfBath, FullBath,HalfBath,BedroomAbvGr,KitchenAbvGr,TotRmsAbvGrd,Fireplaces,GarageCars,PoolArea,MoSold,YrSold,

# In[ ]:




