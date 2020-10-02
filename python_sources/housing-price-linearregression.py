#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats


# In[ ]:


housing = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
housing.head()


# In[ ]:


housing.describe()


# In[ ]:


housing.info()


# In[ ]:


housing.duplicated().sum()


# In[ ]:


housing['Alley'].fillna('NA',inplace=True)


# In[ ]:


housing.drop('Neighborhood',axis=1,inplace=True)


# In[ ]:


sns.scatterplot(housing['GrLivArea'],housing.SalePrice)


# In[ ]:


housing['GrLivArea'] = housing[housing.GrLivArea <=4000]['GrLivArea']


# In[ ]:


sns.scatterplot(housing['GrLivArea'],housing.SalePrice)


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


housing['MSSubClass'] = LabelEncoder().fit_transform(housing['MSSubClass'])


# In[ ]:


housing.head()


# In[ ]:


housing['MSZoning'].fillna('NA',inplace=True)


# In[ ]:


housing['MSZoning'].replace(['FV','RL','RH','RP','RM','C (all)'],['V','R','R','R','R','C'],inplace=True)


# In[ ]:


housing['MSZoning'] = LabelEncoder().fit_transform(housing['MSZoning'])


# In[ ]:


housing['Street'] = LabelEncoder().fit_transform(housing['Street'])


# In[ ]:


housing['Alley'] = LabelEncoder().fit_transform(housing['Alley'])


# In[ ]:


housing['LotShape'].replace(['Reg','IR1','IR2','IR3'],['Reg','Reg','IReg','IReg'],inplace=True)


# In[ ]:


housing['LotShape'] = LabelEncoder().fit_transform(housing['LotShape'])


# In[ ]:


housing['LandContour'] = LabelEncoder().fit_transform(housing['LandContour'])


# In[ ]:


housing['Utilities'].fillna('NA',inplace=True)


# In[ ]:


housing['Utilities'] = LabelEncoder().fit_transform(housing['Utilities'])


# In[ ]:


housing['LotFrontage'].fillna(housing['LotFrontage'].median(),inplace=True)


# In[ ]:


housing['LandSlope'] = LabelEncoder().fit_transform(housing['LandSlope'])


# In[ ]:


housing.drop(['Condition1','Condition2','LotConfig'],axis=1,inplace=True)


# In[ ]:


housing['BldgType'].replace(['1Fam','2FmCon','Duplx','TwnhsE','TwnhsI'],['Fam','Fam','Dup','Twnh','Twnh'],inplace=True)


# In[ ]:


housing['BldgType'] = LabelEncoder().fit_transform(housing['BldgType'])


# In[ ]:


housing['HouseStyle'].replace(['1Story','1.5Fin','1.5Unf','2Story','TwnhsI'],['Fam','Fam','Dup','Twnh','Twnh'],inplace=True)


# In[ ]:


housing['HouseStyle'] = LabelEncoder().fit_transform(housing['HouseStyle'])


# In[ ]:


housing['OverallRating'] = housing['OverallCond'] + housing['OverallQual']


# In[ ]:


housing.drop(['OverallCond','OverallQual'],axis=1,inplace=True)


# In[ ]:


housing.drop(['RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea'],axis=1,inplace=True)


# In[ ]:


housing['ExterQual'] = housing['ExterQual'].map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})


# In[ ]:


housing['ExterCond'] = housing['ExterCond'].map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})


# In[ ]:


housing['ExterRating'] = housing['ExterQual'] + housing['ExterCond']


# In[ ]:


housing.drop(['ExterCond','ExterQual'],axis=1,inplace=True)


# In[ ]:


housing.drop(['Foundation'],axis=1,inplace=True)


# In[ ]:


housing['BsmtQual'].fillna('NA',inplace=True)
housing['BsmtCond'].fillna('NA',inplace=True)
housing['BsmtExposure'].fillna('NA',inplace=True)
housing['BsmtFinType1'].fillna('NA',inplace=True)
housing['BsmtFinType2'].fillna('NA',inplace=True)


# In[ ]:


housing['BsmtQual'] = housing['BsmtQual'].map({'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5,'NA':0})
housing['BsmtCond'] = housing['BsmtCond'].map({'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5,'NA':0})
housing['BsmtExposure'] = housing['BsmtExposure'].map({'No':0,'Mn':1,'Av':2,'Gd':3,'NA':-1})


# In[ ]:


housing['BsmtFinType2'] = housing['BsmtFinType2'].map({'LwQ':1,'Rec':2,'BLQ':3,'ALQ':4,'GLQ':5,'NA':-1,'Unf':0})
housing['BsmtFinType1'] = housing['BsmtFinType1'].map({'LwQ':1,'Rec':2,'BLQ':3,'ALQ':4,'GLQ':5,'NA':-1,'Unf':0})


# In[ ]:


housing['BasementRating'] = housing['BsmtQual'] + housing['BsmtFinType1'] + housing['BsmtCond'] + housing['BsmtExposure'] + housing['BsmtFinType2']


# In[ ]:


housing.drop(['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType2','BsmtFinType1'],axis=1,inplace=True)


# In[ ]:


housing['BasementArea'] = housing['BsmtFinSF1'] + housing['BsmtFinSF2'] + housing['BsmtUnfSF'] + housing['TotalBsmtSF']


# In[ ]:


housing.drop(['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF'],axis=1,inplace=True)


# In[ ]:


housing['Heating'].replace(['GasA','GasB'],['Gas','Gas'],inplace=True)


# In[ ]:


housing['Heating'] = LabelEncoder().fit_transform(housing['Heating'])


# In[ ]:


housing['HeatingQC'] = LabelEncoder().fit_transform(housing['HeatingQC'])


# In[ ]:


housing.drop('Electrical',axis=1,inplace=True)


# In[ ]:


housing['TotalArea'] = housing['1stFlrSF'] + housing['2ndFlrSF'] + housing['LowQualFinSF']


# In[ ]:


housing.drop(['1stFlrSF','2ndFlrSF','LowQualFinSF'],axis=1,inplace=True)


# In[ ]:


housing['TotalBathroom'] = housing['BsmtFullBath'] + 0.5*housing['BsmtHalfBath'] +housing['FullBath'] + 0.5*housing['HalfBath']


# In[ ]:


housing.drop(['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath'],axis=1,inplace=True)


# In[ ]:


housing['KitchenQual'] = housing['KitchenQual'].map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})


# In[ ]:


housing['Functional'].fillna('NA',inplace=True)


# In[ ]:


housing['Functional'] = LabelEncoder().fit_transform(housing['Functional'])


# In[ ]:


housing['FireplaceQu'].fillna('NA',inplace=True)


# In[ ]:


housing['FireplaceQu'] = housing['FireplaceQu'].map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4,'NA':-1})


# In[ ]:


housing['GarageType'].fillna('NA',inplace=True)


# In[ ]:


housing['GarageType'] = LabelEncoder().fit_transform(housing['GarageType'])


# In[ ]:


housing.drop('GarageYrBlt',axis=1,inplace=True)


# In[ ]:


housing['GarageType'] = LabelEncoder().fit_transform(housing['GarageType'])


# In[ ]:


housing['GarageFinish'].fillna('NA',inplace=True)
housing['GarageQual'].fillna('NA',inplace=True)
housing['GarageCond'].fillna('NA',inplace=True)


# In[ ]:


housing['GarageFinish'] = housing['GarageFinish'].map({'Fin':1,'RFn':1,'Unf':1,'NA':0})


# In[ ]:


housing['GarageQual'] = housing['GarageQual'].map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4,'NA':-1})


# In[ ]:


housing['GarageCond'] = housing['GarageCond'].map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4,'NA':-1})


# In[ ]:


housing['GarageRating'] = housing['GarageCond'] + housing['GarageQual']


# In[ ]:


housing.drop(['GarageCond','GarageQual'],axis=1,inplace=True)


# In[ ]:


housing['PavedDrive'] = LabelEncoder().fit_transform(housing['PavedDrive'])


# In[ ]:


housing.drop(['WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch'],axis=1,inplace=True)


# In[ ]:


housing.drop(['PoolArea'],axis=1,inplace=True)


# In[ ]:


housing['PoolQC'].fillna('NA',inplace=True)
housing['Fence'].fillna('NA',inplace=True)


# In[ ]:


housing['PoolQC'] = housing['PoolQC'].map({'Fa':1,'TA':2,'Gd':3,'Ex':4,'NA':0})


# In[ ]:


housing['Fence'] = housing['Fence'].map({'GdPrv':1,'MnPrv':2,'GdWo':3,'MnWw':4,'NA':0})


# In[ ]:


housing['MiscFeature'].fillna('NA',inplace=True)


# In[ ]:


housing['MiscFeature'] = LabelEncoder().fit_transform(housing['MiscFeature'])


# In[ ]:


housing['SaleType'].fillna('NA',inplace=True)


# In[ ]:


housing['SaleType'] = LabelEncoder().fit_transform(housing['SaleType'])


# In[ ]:


housing['SaleCondition'] = LabelEncoder().fit_transform(housing['SaleCondition'])


# In[ ]:


housing.head()


# In[ ]:


housing['GrLivArea'].fillna(housing['GrLivArea'].mean(),inplace=True)


# In[ ]:


housing.drop('CentralAir',axis=1,inplace=True)


# In[ ]:


housing.info()


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lr = LinearRegression()


# In[ ]:


x = housing.drop(['SalePrice','Id','Heating','Fence','GarageRating','LotShape','MoSold','MiscFeature','MiscVal','Alley','GarageArea'],axis=1)
y = housing['SalePrice']


# In[ ]:


lr.fit(x,y)


# In[ ]:


ypred = lr.predict(x)


# In[ ]:


#ypred


# In[ ]:


from sklearn.metrics import r2_score,mean_squared_error


# In[ ]:


r2_score(y,ypred)


# In[ ]:


np.sqrt(mean_squared_error(y,ypred))

