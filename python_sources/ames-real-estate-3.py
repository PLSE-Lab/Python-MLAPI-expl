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


ames=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')


# In[ ]:


ames.head()
ames.info()


# In[ ]:


ames.shape


# In[ ]:


ames.info()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
f,ax=plt.subplots(figsize=(12,9))
_=plt.hist(x='SalePrice',data=ames,bins=50)


# Highly righ skewed data. To fit to a linear model better to take Log of the Slae proice

# In[ ]:


ames['SalePrice']=np.log(ames['SalePrice'])
_=plt.hist(x='SalePrice',data=ames)


# Right Skew is removed. We will use this as the target variable going forward

# ## Data Exploration

# In[ ]:


# Understand the relationship between Gr living area and Sale price
sns.jointplot(x='GrLivArea',y='SalePrice',data=ames,kind='reg')
print(ames.loc[:,['GrLivArea','SalePrice']].corr())
plt.show()


# Correlation between Greater Living Area and the Sale Price is quite high (0.7) and the scatter plot also shows a strong linear relationship

# ### Lets understand the realtionship between neighborhood and sale price

# In[ ]:


f,ax=plt.subplots(figsize=(22,9))
ames_neighborhood=ames.groupby('Neighborhood')['SalePrice'].mean().sort_values(ascending=False)
_=plt.plot(ames_neighborhood,marker='.',linestyle='none')


# Some key clusters emerge amongst neighborhoods
# * NoRidge,StoneBr,NridgHt- Cluster 1 high value
# * BrDale,MeadowV IDOTRR -: Cluster2 low value
# * Everyone else Mid value

# In[ ]:


def NeighborClass(series):
    if series=='NoRidge' or series=='NridgHt' or series=='StoneBr':
        return 'High_Val'
    elif series=='BrDale' or series=='MeadowV' or series=='IDOTRR' or series=='Sawyer' or series=='OldTown' or series=='Edwards':
        return 'Low_val'
    else: return'Mid_val'


# In[ ]:


ames['Neighbor_class']=ames['Neighborhood'].apply(NeighborClass)
ames['Neighbor_class'].value_counts()


# In[ ]:


ames_dummies=pd.get_dummies(data=ames,columns=['Neighbor_class'],drop_first=True)


# In[ ]:


sns.boxplot(x='Neighbor_class',y='SalePrice',data=ames)


# In[ ]:


f,ax=plt.subplots(figsize=(22,9))
groupby_year=ames_dummies.groupby('YearBuilt')['SalePrice'].median()
_=plt.plot(groupby_year,linestyle='none',marker='.')


# *Looks like a somewhat linear relationship especially in the period 1940 onwards *

# In[ ]:


# lets understand the correlations between the variables
corr_mat=ames_dummies.loc[:,['GrLivArea','TotalBsmtSF','GarageArea','1stFlrSF','YearBuilt','2ndFlrSF','GarageCars']].corr()
f,ax=plt.subplots(figsize=(12,9))
sns.heatmap(corr_mat,vmax=0.8,square=True,annot=True)


# In[ ]:


def series_dum(series):
    if series==0:
        return 0
    else: return 1


# In[ ]:


ames_dummies['TotSF']=ames_dummies['GrLivArea']+ames_dummies['2ndFlrSF']+ames_dummies['TotalBsmtSF']


# In[ ]:


ames_dummies['Pool_P']=ames_dummies['PoolArea'].apply(series_dum)


# In[ ]:


ames_dummies['CentralAir_P']=ames_dummies['CentralAir'].map({'Y':1,'N':0})


# In[ ]:


sns.boxplot(x='CentralAir_P',y='SalePrice',data=ames_dummies)
print(ames_dummies['CentralAir_P'].value_counts())
plt.show()


# In[ ]:


ames_dummies.info()


# In[ ]:


ames_short=ames_dummies.loc[:,['TotSF','YearBuilt']]


# In[ ]:





# In[ ]:


X=ames_short.values
print(X)
y=ames['SalePrice'].values
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
param_grid={'max_depth':np.arange(1,6)}
dt=DecisionTreeRegressor()
dt_cv=GridSearchCV(dt,param_grid,cv=5)
dt_cv.fit(X_train,y_train)


# In[ ]:


dt_cv.score(X_test,y_test)


# In[ ]:


test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


test['TotalBsmtSF'].fillna(0,inplace=True)
test['TotalSF']=test['GrLivArea']+test['2ndFlrSF']+test['TotalBsmtSF']
test.info()


# In[ ]:


def tranform(df,col):
    df[col]=df[col].apply(series_dum)
    print(df[col].value_counts())
    return df


# In[ ]:


test=tranform(test,'PoolArea')
test['CentralAir']=test['CentralAir'].map({'Y':1,'N':0})
test.info()


# In[ ]:


test['Neighbor_class']=test_features['Neighborhood'].apply(NeighborClass)
test['Neighbor_class'].value_counts()


# In[ ]:


test_d=pd.get_dummies(data=test,columns=['Neighbor_class'],drop_first=True)
test_d.info()


# In[ ]:


test_features_d=test_d.loc[:,['TotalSF','YearBuilt']]


# In[ ]:


test_features_d.info()


# In[ ]:


X_test2=test_features_d.values
log_predict=dt_cv.predict(X_test2)
sales=np.e**log_predict
sales


# In[ ]:


Id=test['Id']
predictions=pd.DataFrame({'Id':Id,'SalePrice':sales})


# In[ ]:


predictions.head()


# In[ ]:


predictions.to_csv('submission.csv',index=False)

