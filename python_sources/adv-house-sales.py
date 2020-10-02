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


df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
dff = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


df.shape
df.head()


# In[ ]:


df.shape, dff.shape


# In[ ]:


df.columns


# In[ ]:


[ col for col in df.columns if df[col].isnull().sum()>0]


# In[ ]:


[ col for col in dff.columns if dff[col].isnull().sum()>0]


# In[ ]:


for i in df.columns:
    if df[i].isna().sum()>0:
        print(i,df[i].isna().sum())


# In[ ]:


for i in dff.columns:
    if dff[i].isna().sum()>0:
        print(i,dff[i].isna().sum())


# In[ ]:


df.info()


# In[ ]:


dff.info()


# In[ ]:


df = df.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1)
dff = dff.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1)


# In[ ]:


df.LotFrontage.value_counts()


# In[ ]:


df.LotFrontage.isna().sum()


# In[ ]:


for i in df.LotFrontage[df.LotFrontage.isna()==True].index:
    df.LotFrontage[i] = np.random.randint(50,80)


# In[ ]:


for i in dff.LotFrontage[dff.LotFrontage.isna()==True].index:
    dff.LotFrontage[i] = np.random.randint(50,80)


# In[ ]:


df.MasVnrType.value_counts()


# In[ ]:


dff.MasVnrType.value_counts()


# In[ ]:


df.MasVnrType.fillna('None',inplace=True)
dff.MasVnrType.fillna('None',inplace=True)


# In[ ]:


df.MasVnrArea.value_counts()


# In[ ]:


dff.MasVnrArea.value_counts()


# In[ ]:


df.MasVnrArea.fillna(0,inplace=True)
dff.MasVnrArea.fillna(0,inplace=True)


# In[ ]:


df.BsmtCond.value_counts()


# In[ ]:


dff.BsmtCond.value_counts()


# In[ ]:


df.BsmtCond.fillna('TA',inplace=True)
dff.BsmtCond.fillna('TA',inplace=True)


# In[ ]:


df.BsmtQual.value_counts()


# In[ ]:


dff.BsmtQual.value_counts()


# In[ ]:


c=0
for i in df[df.BsmtQual.isna()==True].index:
    c=c+1
    if c%2==0:
        df.BsmtQual[i] = 'TA'
    else:
        df.BsmtQual[i] = 'Gd'


# In[ ]:


c=0
for i in dff[dff.BsmtQual.isna()==True].index:
    c=c+1
    if c%2==0:
        dff.BsmtQual[i] = 'TA'
    else:
        dff.BsmtQual[i] = 'Gd'


# In[ ]:


df.BsmtExposure.value_counts()


# In[ ]:


dff.BsmtExposure.value_counts()


# In[ ]:


df.BsmtExposure.fillna('No',inplace=True)
dff.BsmtExposure.fillna('No',inplace=True)


# In[ ]:


df.BsmtFinType1.value_counts()


# In[ ]:


dff.BsmtFinType1.value_counts()


# In[ ]:


c=0
for i in df[df.BsmtFinType1.isna()==True].index:
    c=c+1
    if c%2==0:
        df.BsmtFinType1[i] = 'Unf'
    else:
        df.BsmtFinType1[i] = 'GLQ'


# In[ ]:


c=0
for i in dff[dff.BsmtFinType1.isna()==True].index:
    c=c+1
    if c%2==0:
        dff.BsmtFinType1[i] = 'Unf'
    else:
        dff.BsmtFinType1[i] = 'GLQ'


# In[ ]:


df.BsmtFinType2.value_counts()


# In[ ]:


dff.BsmtFinType2.value_counts()


# In[ ]:


for i in df[df.BsmtFinType2.isna()==True].index:
        df.BsmtFinType2[i] = 'Unf'


# In[ ]:


for i in dff[dff.BsmtFinType2.isna()==True].index:
        dff.BsmtFinType2[i] = 'Unf'


# In[ ]:


df.Electrical.value_counts()


# In[ ]:


dff.Electrical.value_counts()


# In[ ]:


for i in df[df.Electrical.isna()==True].index:
        df.Electrical[i] = 'SBrkr'


# In[ ]:


for i in dff[dff.Electrical.isna()==True].index:
        dff.Electrical[i] = 'SBrkr'


# In[ ]:


df.GarageCond.value_counts()


# In[ ]:


dff.GarageCond.value_counts()


# In[ ]:


for i in df[df.GarageCond.isna()==True].index:
        df.GarageCond[i] = 'TA'


# In[ ]:


for i in dff[dff.GarageCond.isna()==True].index:
        dff.GarageCond[i] = 'TA'


# In[ ]:


df.GarageQual.value_counts()


# In[ ]:


dff.GarageQual.value_counts()


# In[ ]:


for i in df[df.GarageQual.isna()==True].index:
        df.GarageQual[i] = 'TA'


# In[ ]:


for i in dff[dff.GarageQual.isna()==True].index:
        dff.GarageQual[i] = 'TA'


# In[ ]:


df.GarageType.value_counts()


# In[ ]:


dff.GarageType.value_counts()


# In[ ]:


for i in df[df.GarageType.isna()==True].index:
        df.GarageType[i] = 'Attchd'


# In[ ]:


for i in dff[dff.GarageType.isna()==True].index:
        dff.GarageType[i] = 'Attchd'


# In[ ]:


df.GarageYrBlt.value_counts().head(10)


# In[ ]:


dff.GarageYrBlt.value_counts().head(10)


# In[ ]:


for i in df[df.GarageYrBlt.isna()==True].index:
        df.GarageYrBlt[i] = np.random.randint(2003,2008)


# In[ ]:


for i in dff[dff.GarageYrBlt.isna()==True].index:
        dff.GarageYrBlt[i] = np.random.randint(2003,2008)


# In[ ]:


df.GarageFinish.value_counts()


# In[ ]:


dff.GarageFinish.value_counts()


# In[ ]:


c=0
for i in df[df.GarageFinish.isna()==True].index:
    c=c+1
    if c%2==0:
        df.GarageFinish[i] = 'Unf'
    elif c%3==0:
        df.GarageFinish[i]='RFn'
    else:
        df.GarageFinish[i]='Fin'


# In[ ]:


c=0
for i in dff[dff.GarageFinish.isna()==True].index:
    c=c+1
    if c%2==0:
        dff.GarageFinish[i] = 'Unf'
    elif c%3==0:
        dff.GarageFinish[i]='RFn'
    else:
        dff.GarageFinish[i]='Fin'


# In[ ]:


[ col for col in df.columns if df[col].isnull().sum()>0]


# In[ ]:


[ col for col in dff.columns if dff[col].isnull().sum()>0]


# In[ ]:


dff.MSZoning.value_counts()


# In[ ]:


df.MSZoning.value_counts()


# In[ ]:


for i in dff[dff.MSZoning.isna()==True].index:
        dff.MSZoning[i] = 'RL'


# In[ ]:


dff.BsmtFinSF1.value_counts()


# In[ ]:


df.BsmtFinSF1.value_counts()


# In[ ]:


for i in dff[dff.BsmtFinSF1.isna()==True].index:
        dff.BsmtFinSF1[i] = 0


# In[ ]:


dff.BsmtFinSF1 = dff.BsmtFinSF1.astype('int64')


# In[ ]:


df.BsmtFinSF2.value_counts()


# In[ ]:


for i in dff[dff.BsmtFinSF2.isna()==True].index:
        dff.BsmtFinSF2[i] = 0


# In[ ]:


dff.BsmtFinSF2 = dff.BsmtFinSF2.astype('int64')


# In[ ]:


dff.BsmtUnfSF.value_counts()


# In[ ]:


df.BsmtUnfSF.value_counts()


# In[ ]:


dff.BsmtUnfSF = dff.BsmtUnfSF.astype('int64')


# In[ ]:


for i in dff[dff.BsmtUnfSF.isna()==True].index:
        dff.BsmtUnfSF[i] = 0


# In[ ]:


dff.TotalBsmtSF.value_counts()


# In[ ]:


df.TotalBsmtSF.value_counts()


# In[ ]:


c=0
for i in dff[dff.TotalBsmtSF.isna()==True].index:
    c=c+1
    if c%2==0:
        dff.TotalBsmtSF[i] = 0
    else:
        dff.TotalBsmtSF[i]=864


# In[ ]:


dff.TotalBsmtSF = df.TotalBsmtSF.astype('int64')


# In[ ]:


dff.BsmtFullBath.value_counts()


# In[ ]:


df.BsmtFullBath.value_counts()


# In[ ]:


for i in dff[dff.BsmtFullBath.isna()==True].index:
        dff.BsmtFullBath[i] = 0


# In[ ]:


dff.BsmtFullBath = dff.BsmtFullBath.astype('int64')


# In[ ]:


dff.BsmtHalfBath.value_counts()


# In[ ]:


df.BsmtHalfBath.value_counts()


# In[ ]:


for i in dff[dff.BsmtHalfBath.isna()==True].index:
        dff.BsmtHalfBath[i] = 0


# In[ ]:


dff.BsmtHalfBath = dff.BsmtHalfBath.astype('int64')


# In[ ]:


dff.KitchenQual.value_counts()


# In[ ]:


df.KitchenQual.value_counts()


# In[ ]:


c=0
for i in dff[dff.KitchenQual.isna()==True].index:
    c=c+1
    if c%2==0:
        dff.KitchenQual[i] = 'TA'
    else:
        dff.KitchenQual[i]='Gd'


# In[ ]:


dff.Functional.value_counts()


# In[ ]:


df.Functional.value_counts()


# In[ ]:


for i in dff[dff.Functional.isna()==True].index:
        dff.Functional[i] = 'Typ'


# In[ ]:


dff.GarageCars.value_counts()


# In[ ]:


df.GarageCars.value_counts()


# In[ ]:


for i in dff[dff.GarageCars.isna()==True].index:
        dff.GarageCars[i] = 2


# In[ ]:


dff.GarageCars = dff.GarageCars.astype('int64')


# In[ ]:


dff.SaleType.value_counts()


# In[ ]:


df.SaleType.value_counts()


# In[ ]:


for i in dff[dff.SaleType.isna()==True].index:
        dff.SaleType[i] = 'WD'


# In[ ]:


dff.GarageArea.value_counts()


# In[ ]:


df.GarageArea.value_counts()


# In[ ]:


for i in dff[dff.GarageArea.isna()==True].index:
        dff.GarageArea[i] = 0


# In[ ]:


dff.GarageArea = dff.GarageArea.astype('int64')


# In[ ]:


[col for col in dff.columns if dff[col].isna().sum()>0]


# In[ ]:


df.info()


# In[ ]:


dff.info()


# In[ ]:


dff.Exterior1st.value_counts()


# In[ ]:


df.Exterior1st.value_counts()


# In[ ]:


for i in dff[dff.Exterior1st.isna()==True].index:
        dff.Exterior1st[i] = 'VinylSd'


# In[ ]:


for i in dff[dff.Exterior2nd.isna()==True].index:
        dff.Exterior2nd[i] = 'VinylSd'


# In[ ]:


df.drop('Utilities',axis=1,inplace=True)
dff.drop('Utilities',axis=1,inplace=True)


# In[ ]:


df.iloc[:5,:20]


# In[ ]:


x = df.describe(include='all')


# In[ ]:


df.info()


# In[ ]:


x.iloc[:,60:]


# In[ ]:


df.columns


# In[ ]:


df.drop(['Street','LandContour','LandSlope','Condition1','Condition2','BldgType','RoofMatl','ExterCond','BsmtCond','BsmtFinType2','Heating','CentralAir','Electrical','GarageQual','GarageCond','PavedDrive','SaleType'],axis=1,inplace=True)
dff.drop(['Street','LandContour','LandSlope','Condition1','Condition2','BldgType','RoofMatl','ExterCond','BsmtCond','BsmtFinType2','Heating','CentralAir','Electrical','GarageQual','GarageCond','PavedDrive','SaleType'],axis=1,inplace=True)


# In[ ]:


df = pd.get_dummies(df,drop_first=True)
dff = pd.get_dummies(dff,drop_first=True)


# In[ ]:


df.shape, dff.shape


# In[ ]:


l1 = df.columns
l2 = dff.columns


# In[ ]:


l1 = set(l1)
l2 = set(l2)


# In[ ]:


l1 - l2


# In[ ]:


df.drop(['Exterior1st_ImStucc','Exterior1st_Stone','Exterior2nd_Other','HouseStyle_2.5Fin'],axis=1,inplace=True)


# In[ ]:


df.shape, dff.shape


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


x = df.corr()['SalePrice']


# In[ ]:


x = x.reset_index()


# In[ ]:


x.iloc[:20,1:]


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


# In[ ]:


from sklearn.linear_model import LinearRegression
lr  = LinearRegression(normalize=True)


# In[ ]:


X = df.drop('SalePrice',axis=1)
y = df.SalePrice


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error


# In[ ]:


X_train,X_valid,y_train,y_valid = train_test_split(X,y,random_state=0,test_size=0.2)


# In[ ]:


lr.fit(X_train,y_train)
lr.score(X_train,y_train)


# In[ ]:


lr.score(X_valid,y_valid)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rr = RandomForestRegressor(n_estimators=100,n_jobs=4,random_state=0).fit(X_train,y_train)


# In[ ]:


rr.score(X_train,y_train), rr.score(X_valid,y_valid)


# In[ ]:


from xgboost import XGBRegressor
dd = XGBRegressor(n_estimators=1000,n_jobs=4,learning_rate=0.05).fit(X_train,y_train)


# In[ ]:


dd.score(X_train,y_train), dd.score(X_valid,y_valid)


# In[ ]:


ff = X.assign(const=1)
vif = pd.DataFrame([variance_inflation_factor(ff.values,i) for i in range(ff.shape[1])],index=ff.columns)
vif


# In[ ]:


vif.reset_index(inplace=True)
vif.columns = ['col','val']
vif.head()


# In[ ]:


vif.sort_values(by='val',inplace=True)


# In[ ]:


v = vif[vif.val<=10].col


# In[ ]:


len(v)


# In[ ]:


X1 = df[v]
X_train1,X_valid1,y_train1,y_valid1 = train_test_split(X1,y,test_size=0.2)


# In[ ]:


from xgboost import XGBRegressor
dd1 = XGBRegressor(n_estimators=1000,n_jobs=4).fit(X_train1,y_train1)


# In[ ]:


dd1.score(X_train1,y_train1), dd1.score(X_valid1,y_valid1)


# In[ ]:


rr1 = RandomForestRegressor(n_estimators=1500,random_state=0).fit(X_train1,y_train1)
rr1.score(X_train1,y_train1), rr1.score(X_valid1,y_valid1)


# In[ ]:


lr1 = LinearRegression().fit(X_train1,y_train1)
lr1.score(X_train1,y_train1), lr1.score(X_valid1,y_valid1)


# In[ ]:


dff = dff[v]
pre = dd1.predict(dff)


# In[ ]:


pre.shape


# In[ ]:


dff.Id = dff.Id.astype(int)


# In[ ]:


Submission=pd.DataFrame( { 'Id' : dff['Id'] , 'SalePrice' : pre} )
Submission.to_csv('Submission.csv',index=False)


# In[ ]:




