#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# basic training of the data... nothing complicated.
# I will import libraries when I need them :)
# 

# In[ ]:


data  = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')


# lets have a look at the data. Hmm looks a little complicated than the one I have learned at udemy. There's first time for handling anything and here's mine :)

# In[ ]:


data.head()


# In[ ]:


data.shape


# 
# looks can be deceiving... there are only 1460 rows of data but 81 columns ... may be after cleaning the data i want to use PCA for dimentionality reduction 

# In[ ]:


data.info()


# lets see if we have any null values and missing data. I want to plot a heatmap for null values. Guess I need Seaborn let's import that and plot the data

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


plt.figure(figsize = (16,6))
sns.heatmap(data.isnull(),cmap = 'viridis')


# I want to fill all the null values with some meaningful information, And Id column for sure is not needed for data prediction so I am dropping Id column

# In[ ]:


plt.figure(figsize = (16,6))
sns.heatmap(test.isnull(),cmap = 'viridis')


# In[ ]:


data.drop(['Id'], axis = 1,inplace = True)
data.shape
Id = test.Id
test.drop(['Id'], axis = 1,inplace = True)
test.shape


# we have to predict Saleprice so I am creating X without Saleprice and y = Saleprice. Later I want to split my traing data using train test split 

# In[ ]:


X = data.drop(['SalePrice'],axis = 1)
y = data.SalePrice


# I want to creatre a column AgeofHouse . and I want to delete YrSold, YrRemodeled and YearBuilt
# YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)

# In[ ]:


test.head()


# In[ ]:


AgeofHouse = X.YrSold - X.YearRemodAdd
AgeoftestHouse = test.YrSold -test.YearRemodAdd


# In[ ]:


X = pd.concat([X,AgeofHouse.rename('HouseAge')],axis = 1 )
test = pd.concat([test,AgeoftestHouse.rename('HouseAge')],axis = 1 )


# In[ ]:


X.drop(['YrSold','YearBuilt','YearRemodAdd'],axis = 1, inplace = True)
test.drop(['YrSold','YearBuilt','YearRemodAdd'],axis = 1, inplace = True)


# Plotting correlation map for X

# In[ ]:


corr = X.corr()
plt.figure(figsize=(16, 16))
sns.heatmap(corr, cmap='viridis')


# Too Confusing but I see Dark blue for GarageYrBlt and HouseAge. I wanted to delete if the value is more that 0.95. Lets see 

# In[ ]:


X['GarageYrBlt'].corr(X['HouseAge'])


# 

# In[ ]:


X.columns


# In[ ]:


X.Fence = X.Fence.fillna('NoFence')
X.MiscFeature = X.MiscFeature.fillna('None')
test.Fence = test.Fence.fillna('NoFence')
test.MiscFeature = test.MiscFeature.fillna('None')


# In[ ]:


X.PoolQC = X.PoolQC.fillna('NoPool')# description says
test.PoolQC = test.PoolQC.fillna('NoPool')# description says


# If we observe the above data I think the Garage Colmns with Nan has no garage, As othet Garage features are null too . So  'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond' Should be replaced with None

# In[ ]:


for col in ['GarageType', 'GarageCond','GarageFinish','GarageQual']:
    X[col] = X[col].fillna('none')
for col in ['GarageType', 'GarageCond','GarageFinish','GarageQual']:
    test[col] = test[col].fillna('none')


# In[ ]:



X['GarageYrBlt']=X['GarageYrBlt'].fillna(0)
test['GarageYrBlt']=test['GarageYrBlt'].fillna(0)


# In[ ]:


LF = X.LotFrontage.median()
X['LotFrontage'] = X['LotFrontage'].fillna(LF)
LFt =test.LotFrontage.median()
test['LotFrontage'] = test['LotFrontage'].fillna(LFt)


# In[ ]:


X['Alley'] = X['Alley'].fillna('none')
X['MasVnrType'] = X['MasVnrType'].fillna('none')
X['MasVnrArea'] = X['MasVnrArea'].fillna(0)
X['GarageCars'] = X['GarageCars'].fillna(0)
X['GarageArea'] = X['GarageArea'].fillna(0)
test['Alley'] = test['Alley'].fillna('none')
test['MasVnrType'] = test['MasVnrType'].fillna('none')
test['MasVnrArea'] = test['MasVnrArea'].fillna(0)
test['GarageCars'] = test['GarageCars'].fillna(0)
test['GarageArea'] = test['GarageArea'].fillna(0)


# In[ ]:


for col in ['BsmtQual', 'BsmtCond', 'BsmtExposure','BsmtFinType1','BsmtFinType2']:
    X[col] = X[col].fillna('none')
for col in ['BsmtQual', 'BsmtCond', 'BsmtExposure','BsmtFinType1','BsmtFinType2']:
    test[col] = test[col].fillna('none')


# In[ ]:


X['Electrical'] = X['Electrical'].fillna('none')
X['FireplaceQu'] = X['FireplaceQu'].fillna('none')
test['Electrical'] = test['Electrical'].fillna('none')
test['FireplaceQu'] = test['FireplaceQu'].fillna('none')


# In[ ]:





# In[ ]:


plt.figure(figsize = (16,6))
sns.heatmap(test.isnull(),cmap = 'viridis')


# In[ ]:


col_mask=test.isnull().any(axis=0)
col_mask


# lets group all the columns where data type is Object and use label Encoder

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:





# for now I will type all the object data type variables in a for loop and use Label Encodeing

# In[ ]:


cols = ('MSSubClass','MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig',
        'LandSlope','Neighborhood', 'Condition1','Condition2','BldgType','HouseStyle','RoofStyle',
        'RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation',
        'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC',
        'CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish',
        'GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition') 


# In[ ]:


for i in cols:
    
    LE = LabelEncoder() 
    LE.fit(list(X[i].values)) 
    X[i] = LE.transform(list(X[i].values))
   
for i in cols:
    
    LE = LabelEncoder() 
    LE.fit(list(test[i].values)) 
    test[i] = LE.transform(list(test[i].values))
test.head()


# In[ ]:


from sklearn.linear_model import Ridge


# In[ ]:


rr = Ridge(alpha=10)
rr.fit(X, y)
y_pred = rr.predict(X)
resid = y - y_pred
mean_resid = resid.mean()
std_resid = resid.std()
z = (resid - mean_resid) / std_resid
z = np.array(z)
outliers = np.where(abs(z) > abs(z).std() * 3)[0]
outliers


# In[ ]:


X.drop([ 178,  185,  218,  231,  377,  412,  440,  473,  496,  523,  588,
        608,  628,  632,  664,  666,  688,  691,  769,  774,  803,  898,
       1046, 1169, 1181, 1182, 1243, 1298, 1324, 1423] )
y.drop ([ 178,  185,  218,  231,  377,  412,  440,  473,  496,  523,  588,
        608,  628,  632,  664,  666,  688,  691,  769,  774,  803,  898,
       1046, 1169, 1181, 1182, 1243, 1298, 1324, 1423] )      


# In[ ]:


from sklearn.linear_model import LinearRegression
L1 = LinearRegression()
from sklearn.model_selection import train_test_split
from sklearn import metrics
x_tr,x_ts,y_tr,y_ts = train_test_split(X,y,test_size = 0.3,random_state = 42)


# In[ ]:


L1.fit(x_tr,y_tr)


# In[ ]:


pre = L1.predict(x_ts)


# In[ ]:


metrics.r2_score(pre,y_ts)


# In[ ]:


plt.scatter(pre,y_ts)


# In[ ]:


col_mask=test.isnull().any(axis=0)
col_mask


# In[ ]:


test.info()


# In[ ]:



col_mask=test.isnull().any(axis=0)
col_mask
test = test.fillna(0)


# In[ ]:


col_mask=test.isnull().any(axis=0)
col_mask


# In[ ]:


SalePrice = L1.predict(test)
#test.info()


# In[ ]:


predict_Sales=pd.Series(SalePrice, name = 'SalePrice')
result = pd.concat([Id, predict_Sales], axis=1)
result.to_csv('Housing_Pred.csv',index=False)
result.info()


# In[ ]:





# In[ ]:





# In[ ]:




