#!/usr/bin/env python
# coding: utf-8

# # Kids play with HousingPrice
 First step is to import the required library's:
# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
get_ipython().run_line_magic('matplotlib', 'inline')

To play with data we need to load first ;)
# In[ ]:


T_train = pd.read_csv("../input/train.csv")
T_test = pd.read_csv("../input/test.csv")


# In[ ]:


T_train.columns

We need to concat T_train and T_test and put aside our target attribute SalePrice for some time:
          ......Going to miss u SalePrice :(  
# In[ ]:


train_x = T_train.drop(["SalePrice"], axis = 1)


# In[ ]:


df = pd.concat([train_x,T_test],axis = 0)


# In[ ]:


df.info()


# In[ ]:


df.head()

Now its time to check for null values or missing values.
# In[ ]:


df.isnull().sum().sort_values(ascending=False) ##ascending = false gives high values to low

Droping attributes which have more than 60% null values: Because filling them with mean or median gives no meaning.. 
# In[ ]:


df=df.drop(['Id','MiscFeature','Fence','PoolQC','Alley'],axis=1)   #Including "Id"


# In[ ]:


num_col = df._get_numeric_data()   #going to get only numaric columns or attributes# Missing value imputation:
num_col.info()


# # Missing value imputation:

# In[ ]:


##First let's do for Numarical attributes:

num_nulls = num_col.isnull().sum().sort_values(ascending=False)

In order to fill null data first we need to remove outliers, so lets get that first>>>>>>
# # Outliers treatment:
Uning user defined lamada finction we can know the out liers
# In[ ]:


def var_summary(x):
    return pd.Series([x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(),  x.std(), x.var(), x.min(), x.quantile(0.01), x.quantile(0.05),x.quantile(0.10),x.quantile(0.25),x.quantile(0.50),x.quantile(0.75), x.quantile(0.90),x.quantile(0.95), x.quantile(0.99),x.max()], 
                  index=['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN', 'P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])

num_col.apply(lambda x: var_summary(x)).T

For our better understanding lets consider only quantiles. 
# In[ ]:


def var_summary(x):
    return pd.Series([x.quantile(0.01), x.quantile(0.05),x.quantile(0.10),x.quantile(0.25),x.quantile(0.50),x.quantile(0.75), x.quantile(0.90),x.quantile(0.95), x.quantile(0.99),x.max()], 
                  index=['P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])

num_col.apply(lambda x: var_summary(x)).T

Now check quantiles P1 and P2 if there is much difference then remove P1.
Do same  for quantiles P99 and P100 if there is much difference then remove P100.if you remove p1 then p2 becomes p1 and if we remove p100 then p99 becomes p100.
# In[ ]:


num_col['LotArea']= num_col['LotArea'].clip_upper(num_col['LotArea'].quantile(0.99))
num_col['LotArea']= num_col['LotArea'].clip_upper(num_col['LotArea'].quantile(0.01))
num_col['MasVnrArea']= num_col['MasVnrArea'].clip_upper(num_col['MasVnrArea'].quantile(0.99))
num_col['BsmtFinSF1']= num_col['BsmtFinSF1'].clip_upper(num_col['BsmtFinSF1'].quantile(0.99))
num_col['BsmtFinSF2']= num_col['BsmtFinSF2'].clip_upper(num_col['BsmtFinSF2'].quantile(0.99))
num_col['BsmtUnfSF']= num_col['BsmtUnfSF'].clip_upper(num_col['BsmtUnfSF'].quantile(0.99))
num_col['TotalBsmtSF']= num_col['TotalBsmtSF'].clip_upper(num_col['TotalBsmtSF'].quantile(0.99))
num_col['1stFlrSF']= num_col['1stFlrSF'].clip_upper(num_col['1stFlrSF'].quantile(0.99))
num_col['2ndFlrSF']= num_col['2ndFlrSF'].clip_upper(num_col['2ndFlrSF'].quantile(0.99))
num_col['LowQualFinSF']= num_col['LowQualFinSF'].clip_upper(num_col['LowQualFinSF'].quantile(0.99))
num_col['GrLivArea']= num_col['GrLivArea'].clip_upper(num_col['GrLivArea'].quantile(0.99))
num_col['FullBath']= num_col['FullBath'].clip_upper(num_col['FullBath'].quantile(0.99))
num_col['BedroomAbvGr']= num_col['BedroomAbvGr'].clip_upper(num_col['BedroomAbvGr'].quantile(0.99))
num_col['TotRmsAbvGrd']= num_col['TotRmsAbvGrd'].clip_upper(num_col['TotRmsAbvGrd'].quantile(0.99))
num_col['Fireplaces']= num_col['Fireplaces'].clip_upper(num_col['Fireplaces'].quantile(0.99))
num_col['GarageYrBlt']= num_col['GarageYrBlt'].clip_upper(num_col['GarageYrBlt'].quantile(0.99))
num_col['GarageCars']= num_col['GarageCars'].clip_upper(num_col['GarageCars'].quantile(0.99))
num_col['GarageArea']= num_col['GarageArea'].clip_upper(num_col['GarageArea'].quantile(0.99))
num_col['WoodDeckSF']= num_col['WoodDeckSF'].clip_upper(num_col['WoodDeckSF'].quantile(0.99))
num_col['OpenPorchSF']= num_col['OpenPorchSF'].clip_upper(num_col['OpenPorchSF'].quantile(0.99))
num_col['EnclosedPorch']= num_col['EnclosedPorch'].clip_upper(num_col['EnclosedPorch'].quantile(0.99))
num_col['3SsnPorch']= num_col['3SsnPorch'].clip_upper(num_col['3SsnPorch'].quantile(0.99))
num_col['ScreenPorch']= num_col['ScreenPorch'].clip_upper(num_col['ScreenPorch'].quantile(0.99))
num_col['PoolArea']= num_col['PoolArea'].clip_upper(num_col['PoolArea'].quantile(0.99))
num_col['MiscVal']= num_col['MiscVal'].clip_upper(num_col['MiscVal'].quantile(0.99))


# In[ ]:


num_nulls


# In[ ]:


## Filling null values attributes with mean...

num_col['LotFrontage']=num_col['LotFrontage'].fillna(num_col['LotFrontage'].mean())  
num_col['GarageYrBlt']=num_col['GarageYrBlt'].fillna(num_col['GarageYrBlt'].mean())
num_col['MasVnrArea']=num_col['MasVnrArea'].fillna(num_col['MasVnrArea'].mean())
num_col['BsmtHalfBath']=num_col['BsmtHalfBath'].fillna(num_col['BsmtHalfBath'].mean())
num_col['BsmtFullBath']=num_col['BsmtFullBath'].fillna(num_col['BsmtFullBath'].mean())
num_col['GarageArea']=num_col['GarageArea'].fillna(num_col['GarageArea'].mean())
num_col['BsmtFinSF1']=num_col['BsmtFinSF1'].fillna(num_col['BsmtFinSF1'].mean())
num_col['BsmtFinSF2']=num_col['BsmtFinSF2'].fillna(num_col['BsmtFinSF2'].mean())
num_col['BsmtUnfSF']=num_col['BsmtUnfSF'].fillna(num_col['BsmtUnfSF'].mean())
num_col['TotalBsmtSF']=num_col['TotalBsmtSF'].fillna(num_col['TotalBsmtSF'].mean())
num_col['GarageCars']=num_col['GarageCars'].fillna(num_col['GarageCars'].mean())


# In[ ]:


num_col.isnull().sum().sum()


# In[ ]:


num_col.columns

Now filling null values for categorical
# In[ ]:


dfcat_cols = df.drop(['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
       'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
       'MoSold', 'YrSold'],axis = 1)


# In[ ]:


dfcat_cols.info()


# In[ ]:


dfcat_cols.isnull().sum().sort_values(ascending = False)


# In[ ]:


dfcat_cols.describe().T ##here "T" is transpose


# In[ ]:


dfcat_cols["FireplaceQu"] = dfcat_cols["FireplaceQu"].fillna('Gd')
dfcat_cols["GarageCond"] = dfcat_cols["GarageCond"].fillna('TA')
dfcat_cols["GarageQual"] = dfcat_cols["GarageQual"].fillna('TA')
dfcat_cols["GarageFinish"] = dfcat_cols["GarageFinish"].fillna('Unf')
dfcat_cols["GarageType"] = dfcat_cols["GarageType"].fillna('Attchd')
dfcat_cols["BsmtCond"] = dfcat_cols["BsmtCond"].fillna('TA')
dfcat_cols["BsmtExposure"] = dfcat_cols["BsmtExposure"].fillna('No')
dfcat_cols["BsmtQual"] = dfcat_cols["BsmtQual"].fillna('TA')
dfcat_cols["BsmtFinType2"] = dfcat_cols["BsmtFinType2"].fillna('Unf')
dfcat_cols["BsmtFinType1"] = dfcat_cols["BsmtFinType1"].fillna('Unf')
dfcat_cols["MasVnrType"] = dfcat_cols["MasVnrType"].fillna('None')
dfcat_cols["MSZoning"] = dfcat_cols["MSZoning"].fillna('RL')
dfcat_cols["Utilities"] = dfcat_cols["Utilities"].fillna('AllPub')
dfcat_cols["Functional"] = dfcat_cols["Functional"].fillna('Typ')
dfcat_cols["Electrical"] = dfcat_cols["Electrical"].fillna('SBrkr')
dfcat_cols["KitchenQual"] = dfcat_cols["KitchenQual"].fillna('TA')
dfcat_cols["SaleType"] = dfcat_cols["SaleType"].fillna('WD')
dfcat_cols["Exterior2nd"] = dfcat_cols["Exterior2nd"].fillna('VinylSd')
dfcat_cols["Exterior1st"] = dfcat_cols["Exterior1st"].fillna('VinylSd')


# In[ ]:


dfcat_cols.isnull().sum().any()

Now concat both dfcat_cols and num_col:
# In[ ]:


df1 = pd.concat([num_col,dfcat_cols],axis = 1)


# In[ ]:


# check for null values
df1.isnull().sum().sum()


# In[ ]:


df1.head()


# In[ ]:


df1.info()

Using slicing method we seperate train to train and test to test
# In[ ]:


tt = df1[0:1460:]
test = df1[1461::]


# In[ ]:


tt.isnull().sum().sum()


# In[ ]:


test.isnull().sum().sum()


# In[ ]:


target = T_train.drop(['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition'],axis =1)


# In[ ]:


train = pd.concat([tt,target],axis =1)


# In[ ]:


train.isnull().sum().sum()


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


numaric_cols =train._get_numeric_data()
numaric_cols.columns


# # Finding correlation with respect to SalePrice:(only for num.attr's)

# In[ ]:


corr = numaric_cols.corr()['SalePrice']
corr[np.argsort(corr,axis=0)].sort_values(ascending=False)


# # Heatmap for correlation with respect to SalePrice

# In[ ]:


num_corr=numaric_cols.corr()
plt.subplots(figsize=(13,10))
sns.heatmap(num_corr,square = True)

I an getting top 14 correlated attributes
# In[ ]:


pp = numaric_cols.drop(['MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond',
          'BsmtFinSF1', 'BsmtFinSF2',
       'BsmtUnfSF', '2ndFlrSF', 'LowQualFinSF',
        'BsmtFullBath', 'BsmtHalfBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr',
        'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
       'MoSold', 'YrSold'],axis =1)
cm =pp.corr()
sns.set(font_scale=1.35)
f, ax = plt.subplots(figsize=(10,10))
hm=sns.heatmap(cm, annot = True, vmax =.8)


# In[ ]:


pp.columns


# In[ ]:


nc = pp.rename(columns ={'1stFlrSF':'FirstFlrSF'})
#we rename this because


# In[ ]:


nc.columns

Not only using heat map we can find correlated attributes using "STATS MODEL API"
# In[ ]:


## Using statsmodel.formula.api we'll find the correlation

import statsmodels.formula.api as smf


# In[ ]:


lm=smf.ols('SalePrice~OverallQual+YearBuilt+YearRemodAdd+MasVnrArea+TotalBsmtSF+FirstFlrSF+GrLivArea+FullBath+TotRmsAbvGrd+Fireplaces+GarageYrBlt+GarageCars+GarageArea',nc).fit()


# In[ ]:


lm.summary()


# In[ ]:


lm.pvalues

If p value is less than 0.05 only then we consider those atrributes as correlated with SalePrice
# In[ ]:


nc['intercept'] = lm.params[0]

Using LinearAlzebra also we can find correlation
# In[ ]:


np.linalg.inv(nc[['OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'TotalBsmtSF',
       'FirstFlrSF', 'GrLivArea', 'FullBath', 'Fireplaces',
       'GarageArea']].corr().as_matrix())


# In[ ]:


#should be less than 5 only then we consider those attributes:
np.diag(np.linalg.inv(nc[['OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'TotalBsmtSF',
       'FirstFlrSF', 'GrLivArea', 'FullBath','Fireplaces', 'TotRmsAbvGrd', 'GarageArea']].corr().as_matrix()), 0)


# In[ ]:


#final numarical columns:
finalnum_cols = nc.drop([ "GarageCars", "GarageYrBlt", "TotRmsAbvGrd"],axis =1)
finalnum_cols.columns


# # Now its time deal categorical attributes:

# In[ ]:


cc = train.drop(['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
       'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
       'MoSold', 'YrSold', 'SalePrice'],axis = 1)


# In[ ]:


cc.columns


# In[ ]:


cc.isnull().sum().any()


# In[ ]:


# we should add SalePrice as categorical attributes does not have target(SalePrice)
categorical_col =pd.concat([cc,nc.SalePrice],axis=1)
categorical_col.columns


# In[ ]:


##Now we need to do stats model.api:

import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[ ]:


lm1 = smf.ols('SalePrice ~MSZoning+Street+LotShape+LandContour+Utilities+LotConfig+LandSlope+Neighborhood+Condition1+Condition2+BldgType+HouseStyle+RoofStyle+RoofMatl+Exterior1st+Exterior2nd+MasVnrType+ExterQual+ExterCond+Foundation+BsmtQual+BsmtCond+BsmtExposure+BsmtFinType1+BsmtFinType2+Heating+HeatingQC+CentralAir+Electrical+KitchenQual+Functional+FireplaceQu+GarageType+GarageFinish+GarageQual+GarageCond+PavedDrive+SaleType+SaleCondition', categorical_col).fit()


# In[ ]:


lm1.summary()


# # Doing ANOVA (or)F-test to doubted attributes:

# In[ ]:


import scipy.stats as stats


# # 1.LotShape

# In[ ]:


categorical_col.LotShape.value_counts()


# In[ ]:


s1 = categorical_col.SalePrice[categorical_col.LotShape=="Reg"]
s2 = categorical_col.SalePrice[categorical_col.LotShape=="IR1"]
s3 = categorical_col.SalePrice[categorical_col.LotShape=="IR2"]
s4 = categorical_col.SalePrice[categorical_col.LotShape=="IR3"]


# In[ ]:


stats.f_oneway(s1, s2, s3, s4)


# # 2. LotConfig

# In[ ]:


categorical_col.LotConfig.value_counts()


# In[ ]:


s1 = categorical_col.SalePrice[categorical_col.LotConfig=="Inside"]
s2 = categorical_col.SalePrice[categorical_col.LotConfig=="Corner"]
s3 = categorical_col.SalePrice[categorical_col.LotConfig=="CulDSac"]
s4 = categorical_col.SalePrice[categorical_col.LotConfig=="FR2"]
s5 = categorical_col.SalePrice[categorical_col.LotConfig=="FR3"]


# In[ ]:


stats.f_oneway(s1, s2, s3, s4, s5)


# # 3.BldgType

# In[ ]:


categorical_col.BldgType.value_counts()


# In[ ]:


s1 = categorical_col.SalePrice[categorical_col.BldgType=="1Fam"]
s2 = categorical_col.SalePrice[categorical_col.BldgType=="TwnhsE"]
s3 = categorical_col.SalePrice[categorical_col.BldgType=="Duplex"]
s4 = categorical_col.SalePrice[categorical_col.BldgType=="Twnhs"]
s5 = categorical_col.SalePrice[categorical_col.BldgType=="2fmCon"]


# In[ ]:


stats.f_oneway(s1, s2, s3, s4, s5)


# # 4.HouseStyle

# In[ ]:


categorical_col.HouseStyle.value_counts()


# In[ ]:


s1 = categorical_col.SalePrice[categorical_col.HouseStyle=="1Story"]
s2 = categorical_col.SalePrice[categorical_col.HouseStyle=="2Story"]
s3 = categorical_col.SalePrice[categorical_col.HouseStyle=="1.5Fin"]
s4 = categorical_col.SalePrice[categorical_col.HouseStyle=="SLvl"]
s5 = categorical_col.SalePrice[categorical_col.HouseStyle=="SFoyer"]
s6 = categorical_col.SalePrice[categorical_col.HouseStyle=="1.5Unf"]
s7 = categorical_col.SalePrice[categorical_col.HouseStyle=="2.5Unf"]
s8 = categorical_col.SalePrice[categorical_col.HouseStyle=="2.5Fin"]


# In[ ]:


stats.f_oneway(s1, s2, s3, s4, s5, s6, s7, s8)


# # 5.RoofStyle

# In[ ]:


categorical_col.RoofStyle.value_counts()


# In[ ]:


s1 = categorical_col.SalePrice[categorical_col.RoofStyle=="Gable"]
s2 = categorical_col.SalePrice[categorical_col.RoofStyle=="Hip"]
s3 = categorical_col.SalePrice[categorical_col.RoofStyle=="Flat"]
s4 = categorical_col.SalePrice[categorical_col.RoofStyle=="Gambrel"]
s5 = categorical_col.SalePrice[categorical_col.RoofStyle=="Mansard"]
s6 = categorical_col.SalePrice[categorical_col.RoofStyle=="Shed"]


# In[ ]:


stats.f_oneway(s1, s2, s3, s4, s5, s6)


# # 6.RoofMatl

# In[ ]:


categorical_col.RoofMatl.value_counts()


# In[ ]:


s1 = categorical_col.SalePrice[categorical_col.RoofMatl=="CompShg"]
s2 = categorical_col.SalePrice[categorical_col.RoofMatl=="Tar&Grv"]
s3 = categorical_col.SalePrice[categorical_col.RoofMatl=="WdShngl"]
s4 = categorical_col.SalePrice[categorical_col.RoofMatl=="WdShake"]
s5 = categorical_col.SalePrice[categorical_col.RoofMatl=="Membran"]
s6 = categorical_col.SalePrice[categorical_col.RoofMatl=="Metal"]
s7 = categorical_col.SalePrice[categorical_col.RoofMatl=="ClyTile"]
s8 = categorical_col.SalePrice[categorical_col.RoofMatl=="Roll"]


# In[ ]:


stats.f_oneway(s1, s2, s3, s4, s5,s6,s7,s8)


# # 7.BsmtExposure

# In[ ]:


categorical_col.BsmtExposure.value_counts()


# In[ ]:


s1 = categorical_col.SalePrice[categorical_col.BsmtExposure=="No"]
s2 = categorical_col.SalePrice[categorical_col.BsmtExposure=="Av"]
s3 = categorical_col.SalePrice[categorical_col.BsmtExposure=="Gd"]
s4 = categorical_col.SalePrice[categorical_col.BsmtExposure=="Mn"]


# In[ ]:


stats.f_oneway(s1, s2, s3, s4)


# # 8.GarageFinish

# In[ ]:


categorical_col.GarageFinish.value_counts()


# In[ ]:


s1 = categorical_col.SalePrice[categorical_col.GarageFinish=="Unf"]
s2 = categorical_col.SalePrice[categorical_col.GarageFinish=="RFn"]
s3 = categorical_col.SalePrice[categorical_col.GarageFinish=="Fin"]


# In[ ]:


stats.f_oneway(s1, s2, s3)


# # 9.GarageQual

# In[ ]:


categorical_col.GarageQual.value_counts()


# In[ ]:


s1 = categorical_col.SalePrice[categorical_col.GarageQual=="TA"]
s2 = categorical_col.SalePrice[categorical_col.GarageQual=="Fa"]
s3 = categorical_col.SalePrice[categorical_col.GarageQual=="Gd"]
s4 = categorical_col.SalePrice[categorical_col.GarageQual=="Ex"]
s5 = categorical_col.SalePrice[categorical_col.GarageQual=="Po"]


# In[ ]:


stats.f_oneway(s1, s2, s3, s4, s5)


# # 10.SaleCondition

# In[ ]:


categorical_col.SaleCondition.value_counts()


# In[ ]:


s1 = categorical_col.SalePrice[categorical_col.SaleCondition=="Normal"]
s2 = categorical_col.SalePrice[categorical_col.SaleCondition=="Partial"]
s3 = categorical_col.SalePrice[categorical_col.SaleCondition=="Abnorml"]
s4 = categorical_col.SalePrice[categorical_col.SaleCondition=="Family"]
s5 = categorical_col.SalePrice[categorical_col.SaleCondition=="Alloca"]
s6 = categorical_col.SalePrice[categorical_col.SaleCondition=="AdjLand"]


# In[ ]:


stats.f_oneway(s1, s2, s3, s4, s5,s6)


# In[ ]:


categorical_col.columns


# In[ ]:


finalcategorical_cols = categorical_col.drop([ 'Street', 'LandContour', 'Utilities',
        'LandSlope', 'Condition1', 'Condition2',
          'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterCond', 'Foundation',
        'BsmtCond',  'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'Electrical',
       'Functional', 'GarageType',
        'PavedDrive', 'SaleType', 'SalePrice'],axis =1)
finalcategorical_cols.columns

Now combine both finalcategorical_cols and testcategorical_cols in order to get perfect dimentions.
we do this because some sub classes in test is missing so in order to fill those we cancat both
# In[ ]:


test.columns


# In[ ]:


testcategorical_cols = test.drop(['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
       'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
       'MoSold', 'YrSold', 'Street', 'LandContour', 'Utilities', 'LandSlope',
       'Condition1', 'Condition2', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'ExterCond', 'Foundation', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'Electrical', 'Functional', 'GarageType', 'PavedDrive',
       'SaleType'],axis = 1)


# In[ ]:


testcategorical_cols.columns


# In[ ]:


cat_concat = pd.concat([finalcategorical_cols,testcategorical_cols],axis = 0)


# In[ ]:


cat_concat.info()


# In[ ]:


dummies_concat =  pd.get_dummies(cat_concat, columns =['MSZoning', 'LotShape', 'LotConfig', 'Neighborhood', 'BldgType',
       'HouseStyle', 'RoofStyle', 'RoofMatl', 'ExterQual', 'BsmtQual',
       'BsmtExposure', 'HeatingQC', 'CentralAir', 'KitchenQual', 'FireplaceQu',
       'GarageFinish', 'GarageQual', 'GarageCond', 'SaleCondition'],drop_first =True)


# In[ ]:


dummies_concat.info()

Now put back train into train and test into test:
# In[ ]:


traincat_cols = dummies_concat[0:1460:]
testcat_cols = dummies_concat[1461::]


# In[ ]:


traincat_cols.isnull().sum().sum()


Now concat train final numaric columns and finat categorical columns for model building:
# In[ ]:


final = pd.concat([finalnum_cols,traincat_cols],axis =1)
final.isnull().sum().sum()


# In[ ]:


Final = final.sample(n = 730, random_state = 123)
Final.head(4)


# In[ ]:


Final1x = Final.drop(['SalePrice'], axis= 1)
Final1y = Final.SalePrice


# In[ ]:


Final2 = final.drop(Final.index)
Final2.info()


# In[ ]:


Final2x = Final2.drop(['SalePrice'], axis= 1)
Final2y = Final2.SalePrice


# In[ ]:


from sklearn.cross_validation import train_test_split


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(
        Final1x,
        Final1y,
        test_size=0.20,
        random_state=123)


# In[ ]:


print (len(X_train), len(X_test))


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


linreg = LinearRegression()
linreg.fit(X_train, Y_train)


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(
        Final2x,
        Final2y,
        test_size=0.20,
        random_state=123)


# In[ ]:


y_pred = linreg.predict(X_test)


# In[ ]:


print(y_pred.mean())


# In[ ]:


X_test.columns


# In[ ]:


from sklearn import metrics


# In[ ]:


metrics.r2_score(Y_test, y_pred)


# In[ ]:


rmse = np.sqrt(metrics.mean_squared_error(Y_test, y_pred))
rmse

I did not found r2 (R_square) but it not a big deal you can do it easily because we alredy done preprocessing for test :) ...............................THE END...................................
# In[ ]:




