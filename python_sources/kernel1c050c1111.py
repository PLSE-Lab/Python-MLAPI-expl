#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd


# In[56]:


data=pd.read_csv('../input/train.csv')


# In[57]:


data.describe()


# In[58]:


data.head()


# In[59]:


data=data.drop('MSZoning',axis='columns')


# In[60]:


data.head()


# In[61]:


data=data.drop(['Street','Alley','LotShape','LandContour','Utilities'],axis='columns')


# In[62]:


data.head()


# In[63]:


data=data.drop(['LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType'],axis='columns')


# In[64]:


data.head()


# In[65]:


data=data.drop(['HouseStyle','YearRemodAdd','RoofStyle'],axis='columns')


# In[66]:


data.head()


# In[67]:


data=data.drop(['RoofMatl','Exterior1st','Exterior2nd'],axis='columns')


# In[68]:


data.head()


# In[69]:


data=data.drop(['MasVnrType','MasVnrArea','ExterQual'],axis='columns')
data.head()


# In[70]:


data=data.drop(['ExterCond','Foundation','BsmtQual'],axis='columns')
data.head()


# In[71]:


data=data.drop(['BsmtCond','BsmtExposure','BsmtFinType1'],axis='columns')
data.head()


# In[72]:


data=data.drop(['BsmtFinType2','BsmtFinSF2'],axis='columns')
data.head()


# In[73]:


data=data.drop(['PoolQC','Fence','MiscFeature','MiscVal','MoSold','SaleType','SaleCondition'],axis='columns')
data.head()


# In[74]:


data=data.drop(['GarageCond','PavedDrive','3SsnPorch','ScreenPorch','PoolArea'],axis='columns')
data.head()


# In[75]:


data=data.drop(['GarageYrBlt','GarageCars','GarageArea'],axis='columns')
data.head()


# In[76]:


data=data.drop(['Functional','Fireplaces','FireplaceQu','GarageType','KitchenQual'],axis='columns')
data.head()


# In[77]:


data=data.drop(['BsmtFullBath','BsmtHalfBath','HalfBath','FullBath'],axis='columns')
data.head()


# In[78]:


data=data.drop(['LowQualFinSF','Electrical'],axis='columns')
data.head()


# In[79]:


data=data.drop(['CentralAir','HeatingQC'],axis='columns')
data.head()


# In[80]:


data=data.drop(['MSSubClass'],axis='columns')
data.head()


# In[81]:


data=data.drop(['Heating'],axis='columns')
data.head()


# In[82]:


mean=data.LotFrontage.mean()


# In[83]:


mean


# In[84]:


data.LotFrontage=data.LotFrontage.fillna(mean)


# In[85]:


data.isna().sum()


# In[87]:


data=data.drop(['GarageFinish','GarageQual'],axis='columns')


# In[88]:


data.head()


# In[89]:


data.isna().sum()


# In[90]:


from sklearn.linear_model import LinearRegression


# In[91]:


model=LinearRegression()


# In[92]:


test=pd.read_csv('../input/test.csv')


# In[118]:


y=data.SalePrice


# In[119]:


X=data.drop('SalePrice',axis='columns')


# In[97]:


model.fit(X,y)


# In[105]:


test=test.drop(['MSZoning','Street','LotShape','LandContour','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle'],axis='columns')


# In[106]:


x=data.columns


# In[107]:


x


# In[110]:


y=test.columns


# In[111]:


y


# In[112]:


for i in y:
    if i not in x:
        test=test.drop(i,axis='columns')
        


# In[114]:


test.head()


# In[121]:


X.head()


# In[123]:


test.isna().sum()


# In[125]:


mean1=test.LotFrontage.mean()


# In[126]:


test['LotFrontage']=test.fillna(mean1)


# In[127]:


test.isna().sum()


# In[128]:


mean2=test.BsmtFinSF1.mean()


# In[129]:


test['BsmtFinSF1']=test.fillna(mean2)


# In[130]:


mean=test.BsmtUnfSF.mean()


# In[131]:


test['BsmtUnfSF']=test.fillna(mean)


# In[133]:


mean3=test.TotalBsmtSF.mean()


# In[134]:


test['TotalBsmtSF']=test.fillna(mean3)


# In[135]:


test.isna().sum()


# In[136]:


model.predict(test)

