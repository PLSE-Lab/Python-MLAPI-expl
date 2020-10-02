#!/usr/bin/env python
# coding: utf-8

# # House Price : Feature Extraction Strategy
# ### exploring all the columns and preparing dataset for modeling
# *you find anything helpful, please upvote*

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# collecting data

# In[ ]:


train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train.shape, test.shape


# In[ ]:


train.head()


# In[ ]:


train.isna().sum().sort_values()


# # Filling NaNs

# ### I made a separate kernels for understanding NaNs and choosing best filling NaNs strategy, you can see here :[house-price-fillna-strategy](http://www.kaggle.com/ashishbarvaliya/house-price-fillna-strategy), here i am not gonna repeat.

# In[ ]:


for col in ['Alley','FireplaceQu','Fence','MiscFeature','PoolQC']:
    train[col].fillna('NA', inplace=True)
    test[col].fillna('NA', inplace=True)
    
train['LotFrontage'].fillna(train["LotFrontage"].value_counts().to_frame().index[0], inplace=True)
test['LotFrontage'].fillna(test["LotFrontage"].value_counts().to_frame().index[0], inplace=True)

train[['GarageQual','GarageFinish','GarageYrBlt','GarageType','GarageCond']].isna().head(7)
for col in ['GarageQual','GarageFinish','GarageYrBlt','GarageType','GarageCond']:
    train[col].fillna('NA',inplace=True)
    test[col].fillna('NA',inplace=True)

for col in ['BsmtQual','BsmtCond','BsmtFinType1','BsmtFinType2','BsmtExposure']:
    train[col].fillna('NA',inplace=True)
    test[col].fillna('NA',inplace=True)

train['Electrical'].fillna('SBrkr',inplace=True)

missings = ['GarageCars','GarageArea','KitchenQual','Exterior1st','SaleType','TotalBsmtSF','BsmtUnfSF','Exterior2nd',
            'BsmtFinSF1','BsmtFinSF2','BsmtFullBath','Functional','Utilities','BsmtHalfBath','MSZoning']

numerical=['GarageCars','GarageArea','TotalBsmtSF','BsmtUnfSF','BsmtFinSF1','BsmtFinSF2','BsmtFullBath','BsmtHalfBath']
categorical = ['KitchenQual','Exterior1st','SaleType','Exterior2nd','Functional','Utilities','MSZoning']

# using Imputer class of sklearn libs.
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy='median',axis=0)
imputer.fit(test[numerical] + train[numerical])
test[numerical] = imputer.transform(test[numerical])
train[numerical] = imputer.transform(train[numerical])

for i in categorical:
    train[i].fillna(train[i].value_counts().to_frame().index[0], inplace=True)
    test[i].fillna(test[i].value_counts().to_frame().index[0], inplace=True)    

train[train['MasVnrType'].isna()][['SalePrice','MasVnrType','MasVnrArea']]

train[train['MasVnrType']=='None']['SalePrice'].median()
train[train['MasVnrType']=='BrkFace']['SalePrice'].median()
train[train['MasVnrType']=='Stone']['SalePrice'].median()
train[train['MasVnrType']=='BrkCmn']['SalePrice'].median()

train['MasVnrArea'].fillna(181000,inplace=True)
test['MasVnrArea'].fillna(181000,inplace=True)

train['MasVnrType'].fillna('NA',inplace=True)
test['MasVnrType'].fillna('NA',inplace=True)

print(train.isna().sum().sort_values()[-2:-1])
print(test.isna().sum().sort_values()[-2:-1])


# # Feature extraction

# separating text columns and numerical columns for better understanding.

# In[ ]:


int64 =[]
objects = []
for col in train.columns.tolist():
    if np.dtype(train[col]) == 'int64' or np.dtype(train[col]) == 'float64':
        int64.append(col)
    else:
        objects.append(col)                      #here datatype is 'object'
len(int64), len(objects)        


#  ## 1) Exploring Numerical columns (int64)

# In[ ]:


train[int64].head()


# in the dataset folder one file 'data_description.txt' is available. in the file you can see that below continues columns. i am separating continues and categorical columns. 

# In[ ]:


continues_int64_cols = ['LotArea', 'LotFrontage', 'MasVnrArea','BsmtFinSF2','BsmtFinSF1','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF',
                  'GrLivArea','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal']
categorical_int64_cols=[]
for i in int64:
    if i not in continues_int64_cols:
        categorical_int64_cols.append(i)

print("continues int64 columns",len(continues_int64_cols))
print("categorical int64 columns",len(categorical_int64_cols)) 
continues_int64_cols, categorical_int64_cols


# ### Here some functions of plot created. we might need.

# In[ ]:


def barplot(X,Y):
    plt.figure(figsize=(7,7))
    sns.barplot(x=X, y=Y)
    plt.show()
def scatter(X,Y):
    plt.figure(figsize=(7,7))
    sns.scatterplot(alpha=0.4,x=X, y=Y)
    plt.show()
def hist(X):
    plt.figure(figsize=(7,7))
    sns.distplot(X, bins=40, kde=True)
    plt.show()
def box(X):
    plt.figure(figsize=(3,7))
    sns.boxplot(y=X)
    plt.show() 
def line(X,Y):
    plt.figure(figsize=(7,7))    
    sns.lineplot(x=X, y=Y,color="coral")
    plt.show() 


# for each columns scatter plot will take more space, so used scatter_matrix

# In[ ]:


pd.plotting.scatter_matrix(train[continues_int64_cols[:5]],diagonal='kde', figsize=(10,10))
plt.show()


# all columns looks almost cool, but 'MasVnrArea' and 'BsmtFinSF2' has interesting values, let's explore it
# ### 'MasVnrArea' and 'BsmtFinSF2'

# In[ ]:


# used log to see all small values
hist(np.log(train['MasVnrArea']+1))


# In[ ]:


hist(np.log(train['BsmtFinSF2']+1))


# In[ ]:


print(train['MasVnrArea'].value_counts())
print(train['BsmtFinSF2'].value_counts())


# around 90% values has to 0.0 values so i am converting continues to categotrical columns,for all value > 0.0 in class 1 

# In[ ]:


train['MasVnrArea'] = train['MasVnrArea'].apply(lambda row: 1.0 if row>0.0 else 0.0)
train['BsmtFinSF2'] = train['BsmtFinSF2'].apply(lambda row: 1.0 if row>0.0 else 0.0)


# In[ ]:


binary_cate_int64_cols = []
binary_cate_int64_cols.append('MasVnrArea')
binary_cate_int64_cols.append('BsmtFinSF2')


# In[ ]:


pd.plotting.scatter_matrix(train[continues_int64_cols[5:11]],diagonal='kde', figsize=(10,10))
plt.show()


# here distribution of 'LowQualFinSF' looks similar to above 'MasVnrArea' columns , so used same strategy
# ### 'LowQualFinSF'

# In[ ]:


train['LowQualFinSF'].value_counts()


# In[ ]:


train['LowQualFinSF'] = train['LowQualFinSF'].apply(lambda row: 1.0 if row>0.0 else 0.0)


# In[ ]:


binary_cate_int64_cols.append('LowQualFinSF')


# In[ ]:


pd.plotting.scatter_matrix(train[continues_int64_cols[11:14]],diagonal='kde', figsize=(8,8))
plt.show()


# look good, moveing forward[](http://)

# In[ ]:


pd.plotting.scatter_matrix(train[continues_int64_cols[14:]],diagonal='kde', figsize=(11,11))
plt.show()


# distribution of all columns looks same as 'MasVnrArea', so....

# In[ ]:


for i in continues_int64_cols[14:]:
    train[i] = train[i].apply(lambda row: 1.0 if row>0.0 else 0.0)
    binary_cate_int64_cols.append(i)

for j in binary_cate_int64_cols:
    if j in continues_int64_cols:
        continues_int64_cols.remove(j)        #these special columns removing from the continues_int64_cols
        
print(len(continues_int64_cols))   
print(len(binary_cate_int64_cols))        
continues_int64_cols, binary_cate_int64_cols    


# ### ploting binary_cate_int64_cols

# In[ ]:


fig, axes = plt.subplots(3, 3, figsize=(20,11))
m=0
for i in range(3):
    for j in range(3):
        if m !=8:              # subplots are 9 and columns we have is 8 so ignoring last box, thats why,i apllied this condition
            sns.barplot(train[binary_cate_int64_cols[m]], train['SalePrice'],ax=axes[i,j])
            m+=1
plt.show()


# look cool, moving forward

# In[ ]:


# we changed values of train only, here for test set
for i in binary_cate_int64_cols:
    test[i] = test[i].apply(lambda row: 1.0 if row>0.0 else 0.0)


# In[ ]:


test[binary_cate_int64_cols].head(6)


# ### continues columns part done , now let's exploring categorical of int64 columns

# In[ ]:


train[categorical_int64_cols].head()


# we have tree columns which has values as Year, lets visualize
# ### 'YearBuilt'

# In[ ]:


plt.figure(figsize=(15,7))
test.groupby('YearBuilt')['YearBuilt'].count().plot()
train.groupby('YearBuilt')['YearBuilt'].count().plot()
plt.legend(['test','train'])


# the distribution of years is almost same for test and train sets, so no worry
# ### 'YrSold'

# In[ ]:


plt.figure(figsize=(15,7))
test.groupby('YrSold')['YrSold'].count().plot()
train.groupby('YrSold')['YrSold'].count().plot()
plt.legend(['test','train'])


# values different but in same range so no worry
# ### 'YearRemodAdd'

# In[ ]:


plt.figure(figsize=(15,7))
test.groupby('YearRemodAdd')['YearRemodAdd'].count().plot()
train.groupby('YearRemodAdd')['YearRemodAdd'].count().plot()
plt.legend(['test','train'])


# the distribution of years is almost same for test and train sets, so no worry

# ### visualizing remaining colunns of categorical_int64

# In[ ]:


fig, axes = plt.subplots(4, 3, figsize=(20,15))
sns.barplot(train[categorical_int64_cols[1]], train['SalePrice'],ax=axes[0,0])
sns.barplot(train[categorical_int64_cols[2]], train['SalePrice'],ax=axes[0,1])
sns.barplot(train[categorical_int64_cols[3]], train['SalePrice'],ax=axes[0,2])
sns.barplot(train[categorical_int64_cols[6]], train['SalePrice'],ax=axes[1,0])
sns.barplot(train[categorical_int64_cols[7]], train['SalePrice'],ax=axes[1,1])
sns.barplot(train[categorical_int64_cols[8]], train['SalePrice'],ax=axes[1,2])
sns.barplot(train[categorical_int64_cols[9]], train['SalePrice'],ax=axes[2,0])
sns.barplot(train[categorical_int64_cols[10]], train['SalePrice'],ax=axes[2,1])
sns.barplot(train[categorical_int64_cols[11]], train['SalePrice'],ax=axes[2,2])
sns.barplot(train[categorical_int64_cols[12]], train['SalePrice'],ax=axes[3,0])
sns.barplot(train[categorical_int64_cols[13]], train['SalePrice'],ax=axes[3,1])
sns.barplot(train[categorical_int64_cols[14]], train['SalePrice'],ax=axes[3,2])
plt.show()


# In[ ]:


barplot(train[categorical_int64_cols[15]], train['SalePrice'])


# looks great, lets exploring objects columns(text)
# ## 2) Exploring objects columns(text categorical)

# In[ ]:


train[objects].head()


# In[ ]:


fig, axes = plt.subplots(4, 4, figsize=(20,15))
m=0
for i in range(4):
    for j in range(4):
        sns.barplot(train[objects[m]], train['SalePrice'], ax=axes[i,j])
        m+=1
plt.show()        


# great distribution, moving forward

# In[ ]:


fig, axes = plt.subplots(4, 4, figsize=(20,15))
m=16
for i in range(4):
    for j in range(4):
        sns.barplot(train[objects[m]], train['SalePrice'], ax=axes[i,j])
        m+=1
plt.show()        


# there are 44 columns in objects, thats why i am using subplots to visualize them all

# In[ ]:


ordinal_categorical_cols =[]
ordinal_categorical_cols.extend(['ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','HeatingQC','KitchenQual'])


# * here we can see that , ExterQual','ExterCond'....(see above) are ordinal categorical features so i am gonna used different strategy them.
# 1.  what is ordinal categorical feature ?: [Click Here](http://towardsdatascience.com/understanding-feature-engineering-part-2-categorical-data-f54324193e63)

# In[ ]:


fig, axes = plt.subplots(3, 4, figsize=(20,15))
m=32
for i in range(3):
    for j in range(4):
        sns.barplot(train[objects[m]], train['SalePrice'], ax=axes[i,j])
        m+=1
plt.show()        


# > below are the ordinal feature

# In[ ]:


ordinal_categorical_cols.extend(['FireplaceQu', 'GarageQual','GarageCond','PoolQC'])


# * here 'GarageYrBlt' is contains years , lets visualize

# In[ ]:


plt.figure(figsize=(15,7))
test.groupby('GarageYrBlt')['GarageYrBlt'].count().plot()
train.groupby('GarageYrBlt')['GarageYrBlt'].count().plot()
plt.legend(['test','train'])


# almost same distribution for test and train

# In[ ]:


for i in ordinal_categorical_cols:
    if i in objects:
        objects.remove(i)            # removing ordinal features from the objects
len(objects), len(ordinal_categorical_cols)        


# In[ ]:


print('ordinal categorical cols ',len(ordinal_categorical_cols))
print('continues int64 cols ',len(continues_int64_cols))             
print('numeric categorical int64 cols ',len(categorical_int64_cols))
print('objects(text) categorical ',len(objects) ) 
print('binary int64 categorical ',len(binary_cate_int64_cols) )                   


# * total = 81, means no worry

# In[ ]:


# removinf 'Id' and 'SalePrice'
categorical_int64_cols.remove('Id')
categorical_int64_cols.remove('SalePrice')


# In[ ]:


len(categorical_int64_cols + objects)


# #### categorical_int64_cols contains categorical feature but as int64 datatype(or float)
# #### objects contains categorical feature but as str datatype
# ### OneHot

# In[ ]:


train_objs_num = len(train)
dataset = pd.concat(objs=[train[categorical_int64_cols + objects], test[categorical_int64_cols+ objects]], axis=0)
dataset_preprocessed = pd.get_dummies(dataset.astype(str), drop_first=True)
train_nominal_onehot = dataset_preprocessed[:train_objs_num]
test_nominal_onehot= dataset_preprocessed[train_objs_num:]
train_nominal_onehot.shape, test_nominal_onehot.shape


# I used get_dummies for onehot of all categorical columns 

# In[ ]:


train_nominal_onehot.head()


# In[ ]:


test_nominal_onehot.head()


# In[ ]:


# train[ordinal_categorical_cols].head()
for i in ordinal_categorical_cols:
    print(train[i].value_counts())


#  ## 3) Ordinal Feature 
#        Ex	Excellent (100+ inches)	
#        Gd	Good (90-99 inches)
#        TA	Typical (80-89 inches)
#        Fa	Fair (70-79 inches)
#        Po	Poor (<70 inches
#        NA	No Basement
# ### order : Ex > Gd > TA > Fa > Po > NA 
# #### order for 'BsmtExplosure' : Gd > Av > Mn > No > NA  
#        Gd	Good Exposure
#        Av	Average Exposure (split levels or foyers typically score average or above)	
#        Mn	Mimimum Exposure
#        No	No Exposure
#        NA	No Basement

# In[ ]:


train['BsmtExposure'] = train['BsmtExposure'].map({'Gd':4, 'Av':3, 'Mn':2, 'No':1,'NA':0})
test['BsmtExposure'] = test['BsmtExposure'].map({'Gd':4, 'Av':3, 'Mn':2, 'No':1,'NA':0})

order = {'Ex':5,
        'Gd':4, 
        'TA':3, 
        'Fa':2, 
        'Po':1,
        'NA':0 }
for i in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']:
    train[i] = train[i].map(order)
    test[i] = test[i].map(order)
test[ordinal_categorical_cols].head()         


# In[ ]:


train[ordinal_categorical_cols].head()         


# looks cool. lets combine all the features[](http://)

# In[ ]:


X = pd.concat([train[ordinal_categorical_cols], train[continues_int64_cols], train[binary_cate_int64_cols], train_nominal_onehot], axis=1)
y = train['SalePrice']
test_final = pd.concat([test[ordinal_categorical_cols], test[continues_int64_cols], test[binary_cate_int64_cols], test_nominal_onehot], axis=1)


# In[ ]:


X.shape, y.shape, test_final.shape


# In[ ]:


X.to_csv('new_train.csv',index=False)
test_final.to_csv('new_test.csv',index=False)


# ## Thank you, upvote please
