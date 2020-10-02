#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/train.csv')


# In[ ]:


df.shape


# In[ ]:


df.describe()


# In[ ]:


df.columns


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum(axis=0)


# In[ ]:


df.isnull().sum(axis=1).value_counts()


# ## Saleprice

# In[ ]:


df.SalePrice.hist()


# The sale price has a high variance. We will take a log of the price as the target

# In[ ]:


df['SalePrice'] = np.log(df['SalePrice'])


# In[ ]:


df.SalePrice.hist()


# In[ ]:


df['SalePrice'] = (df['SalePrice'] - df['SalePrice'].mean())/(df['SalePrice'].max() - df['SalePrice'].min())


# ## Lot frontage

# In[ ]:


df.LotFrontage.hist()


# In[ ]:


df.plot('LotFrontage','SalePrice', style='o')


# In[ ]:


df['LotFrontage'] = np.log(df['LotFrontage'])


# In[ ]:


df.LotFrontage.hist(bins=20)


# There seems to be a linear patten of LotFrontage with SalePrice.
# We can fill the missing values either by -
# 1. Regressing these features and fill the missing values with prediction
# 2. Sample from the normal distribution of LotFrontage
# 
# We will follow the second one.

# In[ ]:


LotFrontage_avg = df['LotFrontage'].mean()
LotFrontage_std = df['LotFrontage'].std()
LotFrontage_null_count = df['LotFrontage'].isnull().sum()
    
age_null_random_list = np.random.normal(loc=LotFrontage_avg, scale=LotFrontage_std, size=LotFrontage_null_count)
df['LotFrontage'][np.isnan(df['LotFrontage'])] = age_null_random_list
#dataset['custAge'] = dataset['custAge'].astype(int)


# In[ ]:


df['LotFrontage'] = (df['LotFrontage'] - df['LotFrontage'].mean())/(df['LotFrontage'].max() - df['LotFrontage'].min())


# In[ ]:


df.plot('LotFrontage','SalePrice', style='o')


# In[ ]:


df.LotFrontage.hist()


# ## Alley

# In[ ]:


df.Alley.unique()


# Alley has most of the data missing. We will ignore Alley

# In[ ]:


df.drop('Alley', axis=1, inplace=True)


# ## MasVnrType

# In[ ]:


df['MasVnrType'].value_counts()


# In[ ]:


df.MasVnrType.isnull().sum()


# There are only 8 missing values and majority of the houses have None. We will replace the null values with None and treat None as a seaparate category.

# In[ ]:


df.ix[df.MasVnrType.isnull(), 'MasVnrType'] = 'None'


# In[ ]:


MasVnrType_dummy = pd.get_dummies(df['MasVnrType'], prefix='MasVnrType',drop_first=True)
df = df.join(MasVnrType_dummy)


# In[ ]:


df.boxplot('SalePrice',by='MasVnrType')


# ## MasVnrArea

# In[ ]:


df.MasVnrArea.hist()


# In[ ]:


df.MasVnrArea.apply(lambda x: np.log(1+x)).hist()


# The missing are same as MasVnrType. We will replace with the majority ones = 0

# In[ ]:


df.ix[df.MasVnrArea.isnull(), 'MasVnrArea'] = 0


# In[ ]:


df['MasVnrArea'] = np.log1p(df.MasVnrArea)


# ## BsmtQual

# In[ ]:


df.BsmtQual.unique()


# In[ ]:


df.BsmtQual.value_counts()


# From Data description, the categories are ordered.

# In[ ]:


df.ix[df.BsmtQual.isnull(), 'BsmtQual'] = 'NA'


# In[ ]:


df['BsmtQual'] = df['BsmtQual'].map({'NA':0,'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}).astype('int')


# In[ ]:


df.boxplot('SalePrice',by='BsmtQual')


# ## BsmtCond

# In[ ]:


df.BsmtCond.unique()


# In[ ]:


df.ix[df.BsmtCond.isnull(), 'BsmtCond'] = 'NA'
df['BsmtCond'] = df['BsmtCond'].map({'NA':0,'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}).astype('int')
df.boxplot('SalePrice',by='BsmtCond')


# In[ ]:


df['BsmtCond'].corr(df['BsmtQual'])


# ## BsmtExposure

# In[ ]:


df.BsmtExposure.value_counts()


# In[ ]:


df.ix[df.BsmtExposure.isnull(), 'BsmtExposure'] = 'NA'
df['BsmtExposure'] = df['BsmtExposure'].map({'NA':0,'No':1, 'Mn':2, 'Av':3, 'Gd':4}).astype('int')
df.boxplot('SalePrice',by='BsmtExposure')


# ## BsmtFinType1

# In[ ]:


df.BsmtFinType1.value_counts()


# In[ ]:


df.ix[df.BsmtFinType1.isnull(), 'BsmtFinType1'] = 'NA'
df['BsmtFinType1'] = df['BsmtFinType1'].map({'NA':0,'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6}).astype('int')
df.boxplot('SalePrice',by='BsmtFinType1')


# ## BsmtFinSF1

# In[ ]:


df.BsmtFinSF1.hist()


# In[ ]:


df.BsmtFinSF1.apply(lambda x: np.log(1+x)).hist()


# In[ ]:


df['BsmtFinSF1'] = np.log1p(df.BsmtFinSF1)


# ## BsmtFinType2

# In[ ]:


df.BsmtFinType2.value_counts()


# In[ ]:


df.ix[df.BsmtFinType2.isnull(), 'BsmtFinType2'] = 'NA'
df['BsmtFinType2'] = df['BsmtFinType2'].map({'NA':0,'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6}).astype('int')
df.boxplot('SalePrice',by='BsmtFinType2')


# ## BsmtFinSF2

# In[ ]:


df.BsmtFinSF2.hist()


# In[ ]:


df.BsmtFinSF2.apply(lambda x: np.log(1+x)).hist()


# In[ ]:


df['BsmtFinSF2'] = np.log1p(df.BsmtFinSF2)


# ## Electrical

# In[ ]:


df.Electrical.value_counts()


# In[ ]:


df.ix[df.Electrical.isnull(),'Electrical'] = 'SBrkr'


# In[ ]:


df['Electrical'] = df['Electrical'].map({'Mix':0,'FuseP':1, 'FuseF':2, 'FuseA':3, 'SBrkr':4}).astype('int')
df.boxplot('SalePrice',by='Electrical')


# ## FireplaceQu

# In[ ]:


df.FireplaceQu.unique()


# In[ ]:


df.ix[df.FireplaceQu.isnull(), 'FireplaceQu'] = 'NA'
df['FireplaceQu'] = df['FireplaceQu'].map({'NA':0,'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}).astype('int')
df.boxplot('SalePrice',by='FireplaceQu')


# ## GarageType

# In[ ]:


df.GarageType.value_counts()


# In[ ]:


df.ix[df.GarageType.isnull(), 'GarageType'] = 'NA'
df['GarageType'] = df['GarageType'].map({'NA':0,'Detchd':1, 'CarPort':2, 'BuiltIn':3, 'Basment':4, 'Attchd':5, '2Types':6}).astype('int')
df.boxplot('SalePrice',by='GarageType')


# ## GarageYrBlt

# In[ ]:


df.GarageYrBlt.hist()


# In[ ]:


df[df.GarageYrBlt.isnull()].SalePrice.hist()


# In[ ]:


df.ix[df.GarageYrBlt.isnull(),'GarageYrBlt'] = df.GarageYrBlt.median()


# In[ ]:


df.GarageYrBlt.describe()


# In[ ]:


_,gbins = pd.qcut(df.GarageYrBlt,5,retbins=True)


# In[ ]:


gbins


# In[ ]:


if 0:
    df.ix[df.GarageYrBlt <= 1957,'GarageYrBlt'] = 1
    df.ix[(df.GarageYrBlt > 1957) & (df.GarageYrBlt <=1973),'GarageYrBlt'] = 2
    df.ix[(df.GarageYrBlt > 1973) & (df.GarageYrBlt <=1993),'GarageYrBlt'] = 3
    df.ix[(df.GarageYrBlt > 1993) & (df.GarageYrBlt <=2004),'GarageYrBlt'] = 4
    df.ix[(df.GarageYrBlt > 2004) & (df.GarageYrBlt <=2010),'GarageYrBlt'] = 5
    df.ix[(df.GarageYrBlt > 2010),'GarageYrBlt'] = 6
    df.ix[df.GarageYrBlt.isnull(),'GarageYrBlt'] = 0


# In[ ]:


df['GarageYrBlt'] = df['GarageYrBlt'].astype(int)


# ## GarageFinish

# In[ ]:


df.ix[df.GarageFinish.isnull(), 'GarageFinish'] = 'NA'
df['GarageFinish'] = df['GarageFinish'].map({'NA':0,'Unf':1, 'RFn':2, 'Fin':3}).astype('int')
df.boxplot('SalePrice',by='GarageFinish')


# ## GarageArea

# In[ ]:


df.GarageArea.hist()


# In[ ]:


df['GarageArea'] = np.log1p(df.GarageArea)


# ## GarageQual

# In[ ]:


df.GarageQual.value_counts()


# In[ ]:


df.ix[df.GarageQual.isnull(), 'GarageQual'] = 'NA'
df['GarageQual'] = df['GarageQual'].map({'NA':0,'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}).astype('int')
df.boxplot('SalePrice',by='GarageQual')


# ## GarageCond

# In[ ]:


df.GarageCond.value_counts()


# In[ ]:


df.ix[df.GarageCond.isnull(), 'GarageCond'] = 'NA'
df['GarageCond'] = df['GarageCond'].map({'NA':0,'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}).astype('int')
df.boxplot('SalePrice',by='GarageCond')


# ## PoolQC and PoolArea

# Most of the entries do not have pools. We will ignore these two variables

# In[ ]:


df.PoolArea.hist()


# In[ ]:


df.drop(['PoolQC','PoolArea'], axis=1, inplace=True)


# ## Fence

# In[ ]:


df.Fence.value_counts()


# In[ ]:


df.ix[df.Fence.isnull(),'Fence']='NA'
df['Fence'] = df['Fence'].map({'NA':0,'MnWw':1, 'GdWo':2, 'MnPrv':3, 'GdPrv':4}).astype('int')
df.boxplot('SalePrice',by='Fence')


# In[ ]:


df.boxplot('SalePrice',by='Fence')


# ## MiscFeature

# In[ ]:


df.MiscFeature.value_counts()


# This feature can be ignored

# In[ ]:


df.drop('MiscFeature',axis=1,inplace=True)


# ## MSZoning

# In[ ]:


df.MSZoning.value_counts()


# In[ ]:


df.boxplot('SalePrice',by='MSZoning')


# The test data has 4 missing values. We will replace them by *RL*

# In[ ]:


df.ix[df.MSZoning.isnull(),'MSZoning'] = 'RL'


# In[ ]:


MSZoning_dummy = pd.get_dummies(df['MSZoning'], prefix='MSZoning',drop_first=True)
df = df.join(MSZoning_dummy)


# ## Street

# In[ ]:


df.Street.value_counts()


# In[ ]:


df.boxplot('SalePrice',by='Street')


# In[ ]:


df['Street'] = df['Street'].map({'Pave':0,'Grvl':1}).astype('int')


# ## LotShape

# In[ ]:


df.LotShape.value_counts()


# In[ ]:


df['LotShape'] = df['LotShape'].map({'IR3':0,'IR2':1,'IR1':2, 'Reg':3}).astype('int')


# In[ ]:


df.boxplot('SalePrice',by='LotShape')


# ## LandContour

# In[ ]:


df.LandContour.value_counts()


# In[ ]:


df['LandContour'] = df['LandContour'].map({'Low':0,'HLS':1,'Bnk':2, 'Lvl':3}).astype('int')


# In[ ]:


df.boxplot('SalePrice',by='LandContour')


# ## Utilities

# In[ ]:


df.Utilities.value_counts()


# We will ignore this column.

# In[ ]:


df.drop('Utilities', axis=1, inplace=True)


# ## LotConfig

# In[ ]:


df.LotConfig.value_counts()


# In[ ]:


df.boxplot('SalePrice',by='LotConfig')


# LotConfig does not look important

# In[ ]:


df.drop('LotConfig', axis=1, inplace=True)


# ## LandSlope

# In[ ]:


df.LandSlope.value_counts()


# In[ ]:


df['LandSlope'] = df['LandSlope'].map({'Sev':0,'Mod':1,'Gtl':2}).astype('int')


# In[ ]:


df.boxplot('SalePrice',by='LandSlope')


# ## Neighborhood

# In[ ]:


df.Neighborhood.value_counts()


# In[ ]:


df.boxplot('SalePrice',by='Neighborhood', rot=90)


# In[ ]:


Neighborhood_dummy = pd.get_dummies(df['Neighborhood'], prefix='Neighborhood',drop_first=True)
df = df.join(Neighborhood_dummy)


# ## Condition1

# In[ ]:


df.Condition1.value_counts()


# In[ ]:


df.boxplot('SalePrice',by='Condition1')


# In[ ]:


df.ix[df.Condition1=='PosA','Condition1'] = 'Pos'
df.ix[df.Condition1=='PosN','Condition1'] = 'Pos'
df.ix[df.Condition1=='RRAn','Condition1'] = 'RRNS'
df.ix[df.Condition1=='RRNn','Condition1'] = 'RRNS'
df.ix[df.Condition1=='RRNe','Condition1'] = 'RREW'
df.ix[df.Condition1=='RRAe','Condition1'] = 'RREW'


# In[ ]:


Condition1_dummy = pd.get_dummies(df['Condition1'], prefix='Condition1',drop_first=True)
df = df.join(Condition1_dummy)


# ## Condition2

# In[ ]:


df.Condition2.value_counts()


# We will ignore Condition2

# In[ ]:


df.drop('Condition2',axis=1, inplace=True)


# ## BldgType

# In[ ]:


df.BldgType.value_counts()


# In[ ]:


df.boxplot('SalePrice',by='BldgType')


# In[ ]:


BldgType_dummy = pd.get_dummies(df['BldgType'], prefix='BldgType',drop_first=True)
df = df.join(BldgType_dummy)


# ## HouseStyle

# In[ ]:


df.HouseStyle.value_counts()


# In[ ]:


df.boxplot('SalePrice',by='HouseStyle')


# In[ ]:


HouseStyle_dummy = pd.get_dummies(df['HouseStyle'], prefix='HouseStyle',drop_first=True)
df = df.join(HouseStyle_dummy)


# ## RoofStyle

# In[ ]:


df.RoofStyle.value_counts()


# In[ ]:


df.boxplot('SalePrice',by='RoofStyle')


# In[ ]:


RoofStyle_dummy = pd.get_dummies(df['RoofStyle'], prefix='RoofStyle',drop_first=True)
df = df.join(RoofStyle_dummy)


# ## RoofMat1

# In[ ]:


df.RoofMatl.value_counts()


# We can ignore this column

# In[ ]:


df.drop('RoofMatl', axis=1, inplace=True)


# ## Exterior1st

# In[ ]:


df.Exterior1st.value_counts()


# In[ ]:


df.boxplot('SalePrice',by='Exterior1st', rot=90)


# In[ ]:


df.ix[df.Exterior1st == 'Stone','Exterior1st'] = 'Other'
df.ix[df.Exterior1st == 'BrkComm','Exterior1st'] = 'Other'
df.ix[df.Exterior1st == 'AsphShn','Exterior1st'] = 'Other'
df.ix[df.Exterior1st == 'ImStucc','Exterior1st'] = 'Other'
df.ix[df.Exterior1st == 'CBlock','Exterior1st'] = 'Other'


# In[ ]:


Exterior1st_dummy = pd.get_dummies(df['Exterior1st'], prefix='Exterior1st',drop_first=True)
df = df.join(Exterior1st_dummy)


# ## Exterior2nd

# In[ ]:


df.Exterior2nd.value_counts()


# In[ ]:


df.boxplot('SalePrice',by='Exterior2nd', rot=90)


# In[ ]:


df.ix[df.Exterior2nd == 'Stone','Exterior2nd'] = 'Other'
df.ix[df.Exterior2nd == 'Brk Cmn','Exterior2nd'] = 'Other'
df.ix[df.Exterior2nd == 'AsphShn','Exterior2nd'] = 'Other'
df.ix[df.Exterior2nd == 'ImStucc','Exterior2nd'] = 'Other'
df.ix[df.Exterior2nd == 'CBlock','Exterior2nd'] = 'Other'


# In[ ]:


Exterior2nd_dummy = pd.get_dummies(df['Exterior2nd'], prefix='Exterior2nd',drop_first=True)
df = df.join(Exterior2nd_dummy)


# ## ExterQual

# In[ ]:


df.ExterQual.value_counts()


# In[ ]:


df['ExterQual'] = df['ExterQual'].map({'Po':0,'Fa':1,'TA':2, 'Gd':3, 'Ex':4}).astype('int')


# In[ ]:


df.boxplot('SalePrice',by='ExterQual', rot=90)


# ## ExterCond

# In[ ]:


df.ExterCond.value_counts()


# In[ ]:


df['ExterCond'] = df['ExterCond'].map({'Po':0,'Fa':1,'TA':2, 'Gd':3, 'Ex':4}).astype('int')


# In[ ]:


df.boxplot('SalePrice',by='ExterCond', rot=90)


# ## Foundation

# In[ ]:


df.Foundation.value_counts()


# In[ ]:


df.boxplot('SalePrice',by='Foundation', rot=90)


# In[ ]:


Foundation_dummy = pd.get_dummies(df['Foundation'], prefix='Foundation',drop_first=True)
df = df.join(Foundation_dummy)


# ## Heating

# In[ ]:


df.Heating.value_counts()


# In[ ]:


df.boxplot('SalePrice',by='Heating', rot=90)


# In[ ]:


Heating_dummy = pd.get_dummies(df['Heating'], prefix='Heating',drop_first=True)
df = df.join(Heating_dummy)


# ## HeatingQC

# In[ ]:


df.HeatingQC.value_counts()


# In[ ]:


df['HeatingQC'] = df['HeatingQC'].map({'Po':0,'Fa':1,'TA':2, 'Gd':3, 'Ex':4}).astype('int')


# In[ ]:


df.boxplot('SalePrice',by='HeatingQC', rot=90)


# ## CentralAir

# In[ ]:


df.CentralAir.value_counts()


# In[ ]:


df.boxplot('SalePrice',by='CentralAir', rot=90)


# In[ ]:


df['CentralAir'] = df['CentralAir'].map({'N':0,'Y':1}).astype('int')


# ## KitchenQual

# In[ ]:


df.KitchenQual.value_counts()


# In[ ]:


df['KitchenQual'] = df['KitchenQual'].map({'Po':0,'Fa':1,'TA':2, 'Gd':3, 'Ex':4}).astype('int')


# In[ ]:


df.boxplot('SalePrice',by='KitchenQual', rot=90)


# ## Functional

# In[ ]:


df.Functional.value_counts()


# In[ ]:


df['Functional'] = df['Functional'].map({'Sev':0,'Maj2':1,'Maj1':2, 'Mod':3, 'Min1':4,'Min2':5,'Typ':6}).astype('int')


# In[ ]:


df.boxplot('SalePrice',by='Functional', rot=90)


# ## PavedDrive

# In[ ]:


df.PavedDrive.value_counts()


# In[ ]:


df['PavedDrive'] = df['PavedDrive'].map({'N':0,'P':1,'Y':2}).astype('int')


# In[ ]:


df.boxplot('SalePrice',by='PavedDrive', rot=90)


# ## SaleType

# In[ ]:


df.SaleType.value_counts()


# In[ ]:


df.boxplot('SalePrice',by='SaleType', rot=90)


# In[ ]:


SaleType_dummy = pd.get_dummies(df['SaleType'], prefix='SaleType',drop_first=True)
df = df.join(SaleType_dummy)


# ## SaleCondition

# In[ ]:


df.SaleCondition.value_counts()


# In[ ]:


df.boxplot('SalePrice',by='SaleCondition', rot=90)


# In[ ]:


SaleCondition_dummy = pd.get_dummies(df['SaleCondition'], prefix='SaleCondition',drop_first=True)
df = df.join(SaleCondition_dummy)


# In[ ]:


df[['GarageType', 'GarageCars','GarageArea','GarageQual', 'GarageCond']].corr()


# In[ ]:


df.plot('GarageArea','GarageCond', style='o')


# Garage condition, quality are highly correlated. We can drop quality column

# In[ ]:


df.drop('GarageQual', axis=1, inplace=True)


# ## LotArea

# In[ ]:


df.LotArea.hist()


# In[ ]:


df['LotArea'] = np.log(df['LotArea'])


# ## TotalBsmtSF

# In[ ]:


df.TotalBsmtSF.hist()


# In[ ]:


df['TotalBsmtSF']=np.log1p(df['TotalBsmtSF'])


# ## 1stFlrSF

# In[ ]:


df['1stFlrSF'].hist()


# In[ ]:


df['1stFlrSF']=np.log(df['1stFlrSF'])


# ## 2ndFlrSF

# In[ ]:


df['2ndFlrSF'].hist()


# In[ ]:


df['2ndFlrSF']=np.log1p(df['2ndFlrSF'])


# ## LowQualFinSF

# In[ ]:


df.LowQualFinSF.value_counts()


# Most of the data has 0. We will drop this column

# In[ ]:


df.drop('LowQualFinSF', axis=1, inplace=True)


# ## GrLivArea

# In[ ]:


df.GrLivArea.hist()


# In[ ]:


df['GrLivArea']=np.log(df['GrLivArea'])


# In[ ]:


df.plot('GrLivArea','SalePrice', style='o')


# In[ ]:


df['GrLivArea'] = (df['GrLivArea'] - df['GrLivArea'].mean())/(df['GrLivArea'].max() - df['GrLivArea'].min())


# ## WoodDeckSF

# In[ ]:


df['WoodDeckSF'].hist()


# In[ ]:


df['WoodDeckSF'] = np.log1p(df['WoodDeckSF'])


# ## OpenPorchSF

# In[ ]:


df['OpenPorchSF'].hist()


# In[ ]:


df['OpenPorchSF'] = np.log1p(df['OpenPorchSF'])


# ## EnclosedPorch

# In[ ]:


df.EnclosedPorch.hist()


# In[ ]:


df['EnclosedPorch'] = np.log1p(df['EnclosedPorch'])


# ## 3SsnPorch

# In[ ]:


df['3SsnPorch'].hist()


# In[ ]:


df.plot('3SsnPorch','SalePrice',style='o')


# In[ ]:


df['3SsnPorch'] = np.log1p(df['3SsnPorch'])


# ## ScreenPorch

# In[ ]:


df.ScreenPorch.hist()


# In[ ]:


df['ScreenPorch'] = np.log1p(df['ScreenPorch'])


# ## MiscVal

# In[ ]:


df.MiscVal.value_counts()


# In[ ]:


df.plot('MiscVal','SalePrice',style='o')


# In[ ]:


df['MiscVal'] = np.log1p(df['MiscVal'])


# In[ ]:


df.drop('MiscVal', axis=1, inplace=True)


# ## MoSold

# In[ ]:


df.MoSold.hist()


# In[ ]:


df.boxplot('SalePrice',by='MoSold', rot=90)


# ## YrSold

# In[ ]:


df.YrSold.hist()


# ## KitchenAbvGr

# In[ ]:


df.KitchenAbvGr.value_counts()


# In[ ]:


df.boxplot('SalePrice',by='KitchenAbvGr', rot=90)


# ## Bedroom

# In[ ]:


df.BedroomAbvGr.value_counts()


# In[ ]:


df.boxplot('SalePrice',by='BedroomAbvGr', rot=90)


# In[ ]:


ordinal_columns = ['MSSubClass','LotFrontage','LotArea','LotShape','OverallQual','OverallCond', 'YearBuilt',
                  'YearRemodAdd', 'MasVnrArea','ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                  'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','HeatingQC','1stFlrSF','2ndFlrSF',
                   'GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','KitchenQual',
                  'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageYrBlt','GarageFinish','GarageCars', 'GarageArea',
                  'GarageCond', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'Fence', 'MoSold', 'YrSold']


# In[ ]:


plt.figure(figsize=(10,10))
correlation1 = df[ordinal_columns].corr()
mask = np.zeros_like(correlation1)
indices = np.triu_indices_from(correlation1)
mask[indices] = True
cmap = sns.diverging_palette(220, 8, as_cmap=True)
ax1 =sns.heatmap(correlation1, vmin = -1, vmax = 1,     cmap = cmap, cbar = True)


# In[ ]:


## Model to be continued...


# In[ ]:


'SaleType' in df.columns


# ## Train-test split

# In[ ]:


df.drop('Id', axis=1, inplace=True)


# In[ ]:


df.drop(['MasVnrType', 'MSZoning', 'Neighborhood', 'Condition1', 'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st'], axis=1, inplace=True)
df.drop(['Exterior2nd', 'Foundation', 'Heating', 'SaleType', 'SaleCondition'], axis=1, inplace=True)


# In[ ]:


df.columns


# In[ ]:


from sklearn.model_selection import train_test_split

def cross_validate(Xs, ys):
    X_train, X_test, y_train, y_test = train_test_split(
            Xs, ys, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test


# In[ ]:


train1_X = df.copy()
train1_X.drop('SalePrice', axis=1, inplace=True)
train1_Y = df['SalePrice'].copy()

X_train, X_test, y_train, y_test = cross_validate(train1_X, train1_Y)

X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)


# In[ ]:


X_train.shape


# In[ ]:


y_train.shape


# ## Model development

# ### Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(X_train, y_train)
print(regr.predict(X_test))


# In[ ]:


regr.score(X_test,y_test)


# In[ ]:


import math
print("RMSE: %.2f"
      % math.sqrt(np.mean((regr.predict(X_test) - y_test) ** 2)))


# ## XGBoost

# In[ ]:


from sklearn.metrics import explained_variance_score
import xgboost
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score


# In[ ]:


import xgboost as xg
max_depth = 10
min_child_weight = 10
subsample = 0.5
colsample_bytree = 0.7
objective = 'reg:linear'
num_estimators = 2000
learning_rate = 0.01

#features = df[feature_columns]
#target = df[target_columns]
clf = xg.XGBRegressor(max_depth=max_depth,
                min_child_weight=min_child_weight,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                objective=objective,
                n_estimators=num_estimators,
                learning_rate=learning_rate)
clf.fit(X_train, y_train)


# In[ ]:


print("Training error = ", r2_score(y_train, clf.predict(X_train))) 

print("Testing error = ", r2_score(y_test, clf.predict(X_test))) 


# In[ ]:


from xgboost import plot_importance, to_graphviz
fig = plt.figure(figsize = (14, 20))
ax = fig.add_subplot(111)

colours = plt.cm.Set1(np.linspace(0, 1, 9))

ax = plot_importance(clf, height = 1, color = colours, grid = False,                      importance_type = 'cover', ax = ax);
for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        
ax.set_xlabel('importance score', size = 16);
ax.set_ylabel('features', size = 16);
ax.set_yticklabels(ax.get_yticklabels(), size = 12);
ax.set_title('Ordering of features by importance to the model learnt', size = 20);

