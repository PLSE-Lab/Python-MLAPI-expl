#!/usr/bin/env python
# coding: utf-8

# In[182]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[183]:


x=pd.read_csv('../input/train.csv')
y=pd.read_csv('../input/test.csv')


# In[184]:


y.info()


# In[185]:


from sklearn.preprocessing import LabelEncoder as le
x['Alley']=le().fit_transform(x['Alley'].astype(str))
x=x.drop(['PoolQC','Alley','MiscFeature'],axis=1)
y=y.drop(['PoolQC','Alley','MiscFeature'],axis=1)


# In[28]:


y.shape


# In[186]:


x.LotFrontage.fillna(x.LotFrontage.median(),inplace=True)
y.LotFrontage.fillna(y.LotFrontage.median(),inplace=True)


# In[187]:


x=x.drop(['Fence'],axis=1)
y=y.drop(['Fence'],axis=1)


# In[21]:


x.info()


# In[188]:


for c,d in zip(x.columns,y.columns):
    if x[c].dtype=='object':
        x[c]=le().fit_transform(x[c].astype(str))
        y[d]=le().fit_transform(y[d].astype(str))
    


# In[34]:


y.shape


# In[29]:


x.info()


# In[189]:


x['GarageYrBlt']=x.GarageYrBlt.fillna(x.GarageYrBlt.median())


# In[190]:


y['MSZoning']=le().fit_transform(y['MSZoning'].astype(str))


# In[191]:


y=y.apply(lambda f:f.fillna(f.median()))


# In[49]:


y.info()


# In[192]:


x.dropna(inplace=True)


# In[40]:


y.shape


# In[193]:



x['Exterior']=x.loc[:,'Exterior1st']+x.loc[:,'Exterior2nd']
x['FlrSF']=x.loc[:,'1stFlrSF']+x.loc[:,'2ndFlrSF']
x=x.drop(['Condition1','Condition2','Exterior1st','Exterior2nd','BsmtFinSF1','BsmtUnfSF'],axis=1)

y['Exterior']=y.loc[:,'Exterior1st']+y.loc[:,'Exterior2nd']
y['FlrSF']=y.loc[:,'1stFlrSF']+y.loc[:,'2ndFlrSF']
y=y.drop(['Condition1','Condition2','Exterior1st','Exterior2nd','BsmtFinSF1','BsmtUnfSF'],axis=1)


# In[195]:



x['Bath']=x.loc[:,'BsmtFullBath']+x.loc[:,'FullBath']+x.loc[:,'BsmtHalfBath']+x.loc[:,'HalfBath']
y['Bath']=y.loc[:,'BsmtFullBath']+y.loc[:,'FullBath']+y.loc[:,'BsmtHalfBath']+y.loc[:,'HalfBath']
x['Sold']=x.loc[:,'YrSold']+(x.loc[:,'MoSold']/12).astype(float)
x['Porch']=x.loc[:,'OpenPorchSF']+x.loc[:,'EnclosedPorch']+x.loc[:,'3SsnPorch']+x.loc[:,'ScreenPorch']
y['Porch']=y.loc[:,'OpenPorchSF']+y.loc[:,'EnclosedPorch']+y.loc[:,'3SsnPorch']+y.loc[:,'ScreenPorch']
y['Sold']=y.loc[:,'YrSold']+(y.loc[:,'MoSold']/12).astype(float)
x=x.drop(['BsmtExposure','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','YrSold','MoSold','MiscVal','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch'],axis=1)
y=y.drop(['BsmtExposure','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','YrSold','MoSold','MiscVal','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch'],axis=1)


# In[196]:


x=x.drop(['GarageCars','GarageFinish','BedroomAbvGr','KitchenAbvGr','1stFlrSF','2ndFlrSF'],axis=1)
y=y.drop(['GarageCars','GarageFinish','KitchenAbvGr','BedroomAbvGr','1stFlrSF','2ndFlrSF'],axis=1)


# In[197]:


x=x.drop(['GarageYrBlt'],axis=1)
y=y.drop(['GarageYrBlt'],axis=1)


# In[198]:


x=x.drop(['Id'],axis=1)
y=y.drop(['Id'],axis=1)


# In[199]:


q=x.corr()


# In[200]:


x['SalePrice']=np.log(x['SalePrice'])


# In[201]:


def match(m):
    if m>0.75 or m<-0.75:
        return 1
    else:
        return 0
p=q.applymap(match)
p=p.values.reshape(x.shape[1],x.shape[1])


# In[202]:


j=zip(*np.where(p==1))
l=[]
for a,b in j:
    if a!=b:
        l.append((a,b))
print(l)


# In[203]:


q=[]
for a,b in l:
    q.append((list(x.columns)[a],list(x.columns)[b]))
print(q)


# In[125]:


x.corr().loc[:,['SalePrice','OverallQual','GrLivArea','TotRmsAbvGrd','FlrSF']]


# In[204]:


x=x.drop(['FlrSF','GrLivArea'],axis=1)
y=y.drop(['FlrSF','GrLivArea'],axis=1)


# In[206]:


from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from xgboost.sklearn import XGBRegressor


# In[211]:


param1={'alpha':[1,5,10,20,30],'l1_ratio':[0.5,0.7,0.9,0.3,2,5,10]}
param2={'criterion':['mse','mae']}
param3={'learning_rate':[0.1,0.2,0.25,0.4,0.5],'n_estimators':[10,50,100,75]}


# In[208]:


z=x['SalePrice']
x=x.drop(['SalePrice'],axis=1)


# In[212]:


p=GridSearchCV(XGBRegressor(),param3)


# In[213]:


p.fit(x,z)


# In[215]:


p.best_params_


# In[216]:


q=p.predict(y[x.columns])


# In[148]:


a=pd.read_csv('../input/test.csv')
s=pd.DataFrame(np.exp(q),index=a['Id'],columns=['SalePrice'])


# In[149]:


s.to_csv('result.csv')


# In[ ]:




