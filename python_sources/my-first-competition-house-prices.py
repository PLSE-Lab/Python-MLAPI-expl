#!/usr/bin/env python
# coding: utf-8

# This is a basic kernel which includes some basic methods for Preprocessing 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Load the train data in a dataframe
train = pd.read_csv("../input/train.csv")

# Load the test data in a dataframe
test = pd.read_csv("../input/test.csv")


# In[ ]:


# Look at the head of the train dataframe
train.head()


# In[ ]:


train.info()


# In[ ]:


# Look at the SalePrice Variable
sns.distplot(train.SalePrice)


# In[ ]:


train.SalePrice.describe()


# **MISSING VALUES TREATMENT**

# In[ ]:


nulls = train.isnull().sum().sort_values(ascending=False)
nulls.head(20)


# > From the above dataframe -'nulls', we came to know that the attributes PoolQC,MiscFeature,Alley and Fence are having morethan 60% of the values as 'nan'.so, its better to remove them as these columns won't give much info about the SalePrice.

# In[ ]:


train = train.drop(['Id','PoolQC','MiscFeature','Alley','Fence'],axis = 1)


# **FireplaceQu**

# In[ ]:


train[['Fireplaces','FireplaceQu']].head(10)


# The attribute 'FireplaceQu' is having 690 null values.If we compare the columns 'FireplaceQu' and 'Fireplaces', the observations which are having the zeros in the Fireplaces column are having the 'nan' values in FireplaceQu. It tells that the houses which are not having the Fireplaces are having nan values in FireplaceQu so, i will replace these nulls with "no Fireplace" i,e 'NF'.

# In[ ]:


train['FireplaceQu'].isnull().sum()


# In[ ]:


train['Fireplaces'].value_counts()


# In[ ]:


# I used fillna() to replace the nulls with NF
train['FireplaceQu']=train['FireplaceQu'].fillna('NF')


# In[ ]:


#LotFrontage
train['LotFrontage'] =train['LotFrontage'].fillna(value=train['LotFrontage'].mean())


# **Attributes related to "GARAGE"**

# In[ ]:


train['GarageType'].isnull().sum()


# In[ ]:


train['GarageCond'].isnull().sum()


# In[ ]:


train['GarageFinish'].isnull().sum()


# In[ ]:


train['GarageYrBlt'].isnull().sum()


# In[ ]:


train['GarageQual'].isnull().sum()


# In[ ]:


train['GarageArea'].value_counts().head()


# We can observe that all the columns related to Garage are having the sama number of null values. so, there should be a relationship among them and if we look at the 'GarageArea' column it is having the 81 zeros which is equal to no: of 'nans' in these columns.Hence we can conclude that the houses without Garage Area are having 'nan' at all these columns.
# >>>I will replace these nans with 'No GarageArea'----> 'NG'

# In[ ]:


train['GarageType']=train['GarageType'].fillna('NG')
train['GarageCond']=train['GarageCond'].fillna('NG')
train['GarageFinish']=train['GarageFinish'].fillna('NG')
train['GarageYrBlt']=train['GarageYrBlt'].fillna('NG')
train['GarageQual']=train['GarageQual'].fillna('NG')


# In[ ]:


#Similarly for the attributes of Bsmt I'll replace with NB
train['BsmtExposure']=train['BsmtExposure'].fillna('NB')
train['BsmtFinType2']=train['BsmtFinType2'].fillna('NB')
train['BsmtFinType1']=train['BsmtFinType1'].fillna('NB')
train['BsmtCond']=train['BsmtCond'].fillna('NB')
train['BsmtQual']=train['BsmtQual'].fillna('NB')


# In[ ]:


# MasVnr
train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mean())
train['MasVnrType'] = train['MasVnrType'].fillna('none')


# In[ ]:


train.Electrical = train.Electrical.fillna('SBrkr')


# In[ ]:


#confirm that the train doesn't have any null values
train.isnull().sum().sum()


# In[ ]:





# ****OUT LIARS****

# In[ ]:


num_train = train._get_numeric_data()


# In[ ]:


# I'll write a pre defined function to look into the outliars with the help of percentiles.
def var_summary(x):
    return pd.Series([x.mean(), x.median(), x.min(), x.quantile(0.01), x.quantile(0.05),x.quantile(0.10),x.quantile(0.25),x.quantile(0.50),x.quantile(0.75), x.quantile(0.90),x.quantile(0.95), x.quantile(0.99),x.max()], 
                  index=['MEAN','MEDIAN', 'MIN', 'P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])

num_train.apply(lambda x: var_summary(x)).T


# From the above data, we can observe the values for 1st ,25th,50th,75th Percentiles along with the max and min values and accordingly i'll remove the Outliars

# In[ ]:


# Boxplot
sns.boxplot(train.LotArea)


# In[ ]:


# Clipping the values which are above 95-pecentile
train['LotArea']= train['LotArea'].clip_upper(train['LotArea'].quantile(0.95)) 


# In[ ]:


sns.boxplot(train.MasVnrArea)


# In[ ]:


train['MasVnrArea']= train['MasVnrArea'].clip_upper(train['MasVnrArea'].quantile(0.95)) 


# In[ ]:


sns.boxplot(train.BsmtFinSF1)


# In[ ]:


train['BsmtFinSF1']= train['BsmtFinSF1'].clip_upper(train['BsmtFinSF1'].quantile(0.95)) 


# In[ ]:


sns.boxplot(train.BsmtFinSF2)


# In[ ]:


train['BsmtFinSF2']= train['BsmtFinSF2'].clip_upper(train['BsmtFinSF2'].quantile(0.99)) 


# In[ ]:


train['BsmtUnfSF']= train['BsmtUnfSF'].clip_upper(train['BsmtUnfSF'].quantile(0.99)) 
train['TotalBsmtSF']= train['TotalBsmtSF'].clip_upper(train['TotalBsmtSF'].quantile(0.99)) 
train['1stFlrSF']= train['1stFlrSF'].clip_upper(train['1stFlrSF'].quantile(0.99)) 
train['2ndFlrSF']= train['2ndFlrSF'].clip_upper(train['2ndFlrSF'].quantile(0.99)) 
train['LowQualFinSF']= train['LowQualFinSF'].clip_upper(train['LowQualFinSF'].quantile(0.99)) 
train['GrLivArea']= train['GrLivArea'].clip_upper(train['GrLivArea'].quantile(0.99)) 
train['PoolArea']= train['PoolArea'].clip_upper(train['PoolArea'].quantile(0.99)) 
train['MiscVal']= train['MiscVal'].clip_upper(train['MiscVal'].quantile(0.99)) 


# In[ ]:


sns.boxplot(train.SalePrice)


# In[ ]:


train['SalePrice']= train['SalePrice'].clip_upper(train['SalePrice'].quantile(0.99)) 


# **Checking for Correlation**

# In[ ]:


# CORREALATION
correlation = num_train .corr()
plt.subplots(figsize=(10,10))
sns.heatmap(correlation,vmax =.8 ,square = True)


# In[ ]:


# Look for highly correlated variables
k = 12
cols = correlation.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(num_train[cols].values.T)
sns.set(font_scale=1.35)
f, ax = plt.subplots(figsize=(10,8))
hm=sns.heatmap(cm, annot = True,vmax =.8, yticklabels=cols.values, xticklabels = cols.values)


# **FEATURE SELECTION**

# In[ ]:


# Dummifyiny the categorical data
dum_train = pd.get_dummies(train)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


dum_train_x = dum_train.drop(["SalePrice"],axis = 1)
dum_train_y = dum_train.SalePrice

X_train = dum_train_x
Y_train = dum_train_y


# In[ ]:


radm_clf = RandomForestRegressor(oob_score=True,n_estimators=100 )
radm_clf.fit( X_train, Y_train )


# In[ ]:


indices = np.argsort(radm_clf.feature_importances_)[::-1]


# In[ ]:


indices = np.argsort(radm_clf.feature_importances_)[::-1]
feature_rank = pd.DataFrame( columns = ['rank', 'feature', 'importance'] )
for f in range(X_train.shape[1]):
    feature_rank.loc[f] = [f+1,
                         X_train.columns[indices[f]],
                         radm_clf.feature_importances_[indices[f]]]
f, ax = plt.subplots(figsize=(10,100))
sns.barplot( y = 'feature', x = 'importance', data = feature_rank, color = 'Yellow')
plt.show()


# In[ ]:


best_train = feature_rank.head(50)
best_train

