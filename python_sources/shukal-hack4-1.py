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
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')
from matplotlib.artist import setp


# In[ ]:


df_train=pd.read_csv('/kaggle/input/predict-the-housing-price/train.csv')


# In[ ]:


df_test=pd.read_csv('/kaggle/input/predict-the-housing-price/Test.csv')
df_test.columns


# In[ ]:


df_train.head()
df_train.columns


# In[ ]:


train=df_train.drop(['Condition2','Foundation','FireplaceQu','PavedDrive','MiscFeature','Fence','PoolQC','Alley'],axis=1)


# In[ ]:


test=df_test.drop(['Condition2','Foundation','FireplaceQu','PavedDrive','MiscFeature','Fence','PoolQC','Alley'],axis=1)


# In[ ]:


train.head()


# In[ ]:



train.shape


# In[ ]:


train=train.fillna(method='ffill')


# In[ ]:


test=test.fillna(method='ffill')


# In[ ]:


train.isnull().any()


# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(data=train,x='MSSubClass')


# In[ ]:


sns.barplot(train.OverallQual, train.SalePrice)


# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(data=train,x='YearBuilt')
plt.xticks(rotation=90)


# In[ ]:


fig = plt.figure(figsize=(15,10))
ax1 = plt.subplot2grid((2,2),(0,0))
plt.scatter(x=train['GrLivArea'], y=train['SalePrice'], color=('yellowgreen'))
plt.axvline(x=4600, color='r', linestyle='-')
plt.title('Ground living Area- Price scatter plot', fontsize=15, weight='bold' )

ax1 = plt.subplot2grid((2,2),(0,1))
plt.scatter(x=train['TotalBsmtSF'], y=train['SalePrice'], color=('red'))
plt.axvline(x=5900, color='r', linestyle='-')
plt.title('Basement Area - Price scatter plot', fontsize=15, weight='bold' )

ax1 = plt.subplot2grid((2,2),(1,0))
plt.scatter(x=train['1stFlrSF'], y=train['SalePrice'], color=('deepskyblue'))
plt.axvline(x=4000, color='r', linestyle='-')
plt.title('First floor Area - Price scatter plot', fontsize=15, weight='bold' )

ax1 = plt.subplot2grid((2,2),(1,1))
plt.scatter(x=train['MasVnrArea'], y=train['SalePrice'], color=('gold'))
plt.axvline(x=1500, color='r', linestyle='-')
plt.title('Masonry veneer Area - Price scatter plot', fontsize=15, weight='bold' )


# In[ ]:


plt.figure(figsize=(20,10))
sns.distplot(train['SalePrice'],color ='b')


# In[ ]:


train.head()


# In[ ]:


train['MSZoning'].unique()
MSZoning_map={'RL':0,'RM':1,'C(all)':2,'FV':3,'RH':4}
for data in train,test:
    data['MSZoning']=data['MSZoning'].map(MSZoning_map)
    data['MSZoning']=data['MSZoning'].fillna(0)


# In[ ]:


train['Street'].unique()
street_map={'Pave':0,'Grvl':1}
for data in train,test:
    data['Street']=data['Street'].map(street_map)


# In[ ]:


train['LotShape'].unique()
Lotshape_map={'Reg':0,'IR1':1,'IR2':2,'IR3':3}
for data in test,train:
    data['LotShape']= data['LotShape'].map(Lotshape_map)


# In[ ]:


train['LandContour'].unique()
landcontour_map={'Lvl':0,'Bnk':1,'Low':2,'HLS':3}
for data in train,test:
    data['LandContour']= data['LandContour'].map(landcontour_map)


# In[ ]:


attributes_train = ['SalePrice','Street','LotShape','LandContour', 'MSSubClass', 'MSZoning', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF','1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'LotArea', 'GarageCars', 'GarageArea', 'EnclosedPorch']
attributes_test =  ['MSSubClass', 'Street','LotShape','LandContour','MSZoning', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'LotArea', 'GarageCars', 'GarageArea', 'EnclosedPorch']
train = train[attributes_train]
test =test[attributes_test]


# In[ ]:


X=train.drop(['SalePrice'],axis=1)
y=train['SalePrice']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42,test_size=0.429)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor


# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


grad_boast=GradientBoostingRegressor()


# In[ ]:


grad_boast.fit(X_train,y_train)


# In[ ]:


y_pred=grad_boast.predict(X_test)
y_pred


# In[ ]:


grad_boast.score(X_train,y_train)


# In[ ]:


rand=RandomForestRegressor()
rand.fit(X_train,y_train)
rand.score(X_train,y_train)


# In[ ]:


submission = pd.DataFrame({
        "Id":df_test['Id'],
        "SalePrice":y_pred
    })
submission.to_csv('shukal1submission.csv',index=False,header=1 )


# In[ ]:




