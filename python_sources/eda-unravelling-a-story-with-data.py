#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis(EDA) is an approach to analyse the data , to summarize its characteristics , often with visual methods. Every machine learning problem solving starts with EDA. It is probably the most important part of a machine learning project.
# 

# [![download.png](attachment:download.png)](http://)

# # Import necessary libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import seaborn as sns


# # Overview of train, test data and target variable

# In[ ]:


#Reading both datasets
train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


# Numeric features
numeric_features=train.select_dtypes(include=[np.number])
numeric_features.dtypes
# Note MSSubClass,OverallQual and OverallCond are actually categorical features in numeric values.


# In[ ]:


# Categorical features
categoricals=train.select_dtypes(exclude=[np.number])
categoricals.dtypes


# In[ ]:


# Target variable
train.SalePrice.describe()


# In[ ]:


# Lets have a look at target/Y variable
plt.hist(train.SalePrice,color='blue')
plt.show()
#The data is skewed, hence to normalise it we will use log.


# In[ ]:


# Log of Y variable
targets=np.log(train.SalePrice)
plt.hist(targets,color='blue')
plt.show()
#Now the data is normal distributed, we will use log data for training the model .


# In[ ]:


#Lets combine both train and test datasets.
train_copy=train.copy()
train_copy.drop(['SalePrice'],axis=1,inplace=True)
combined=train_copy.append(test)
combined.reset_index(inplace=True)
combined.drop(['index','Id'],axis=1,inplace=True)


# In[ ]:


#Null values using heatmap
sns.heatmap(combined.isnull(), cbar=False)


# In[ ]:


# Correlation
corr = numeric_features.corr()
print(corr['SalePrice'].sort_values(ascending=False))


# In[ ]:


# Lets visualize correlation with heatmap
plt.subplots(figsize=(20, 9))
sns.heatmap(corr, square=True)


# # Multicollinearity observed between
# ### a)GarageYrBlt & YearBuilt
# ### b)TotRoomsAbvGrd & GrLivArea
# ### c)GarageArea & GarageCars
# ### d)1stFloorSF & TotalBsmtSF

# # Detailed data analysis - Analysing discrete numeric variables

# In[ ]:


# YearBuilt and GarageYrBlt
plt.scatter(x=combined['YearBuilt'],y=combined['GarageYrBlt'])
plt.xlabel('YearBuilt')
plt.ylabel('GarageYrBlt')
plt.show()
#Note that in some cases YrBuilt is more than GarageYrBlt, which is impossible! 
#Hence we can assume that there was an error in filling the data. Also one point is clearly an outlier. 


# In[ ]:


#Quality vs SalePrice - High correlation
quality_pivot = train.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)
quality_pivot.plot(kind='bar',color='blue')
plt.xlabel('OverallQual')
plt.ylabel('SalePrice')
plt.show()


# In[ ]:


#OverallCond vs SalePrice - Not much correlation
cond_pivot = train.pivot_table(index='OverallCond', values='SalePrice', aggfunc=np.median)
cond_pivot.plot(kind='bar',color='blue')
plt.xlabel('OverallCond')
plt.ylabel('SalePrice')
plt.show()


# In[ ]:


#HalfBath vs SalePrice
halfbath_pivot = train.pivot_table(index='HalfBath', values='SalePrice', aggfunc=np.median)
halfbath_pivot.plot(kind='bar',color='blue')
plt.xlabel('HalfBath')
plt.ylabel('SalePrice')
plt.show()


# In[ ]:


#FullBath vs SalePrice
fullbath_pivot = train.pivot_table(index='FullBath', values='SalePrice', aggfunc=np.median)
fullbath_pivot.plot(kind='bar',color='blue')
plt.xlabel('FullBath')
plt.ylabel('SalePrice')
plt.show()


# In[ ]:


#BsmtFullBath vs SalePrice
BsmtFullBath_pivot = train.pivot_table(index='BsmtFullBath', values='SalePrice', aggfunc=np.median)
BsmtFullBath_pivot.plot(kind='bar',color='blue')
plt.xlabel('BsmtFullBath')
plt.ylabel('SalePrice')
plt.show()


# In[ ]:


#BsmtHalfBath vs SalePrice
BsmtHalfBath_pivot = train.pivot_table(index='BsmtHalfBath', values='SalePrice', aggfunc=np.median)
BsmtHalfBath_pivot.plot(kind='bar',color='blue')
plt.xlabel('BsmtHalfBath')
plt.ylabel('SalePrice')
plt.show()


# In[ ]:


#BedroomAbvGr vs SalesPrice
BedroomAbvGr_pivot = train.pivot_table(index='BedroomAbvGr', values='SalePrice', aggfunc=np.median)
BedroomAbvGr_pivot.plot(kind='bar',color='blue')
plt.xlabel('BedroomAbvGr')
plt.ylabel('SalePrice')
plt.show()


# In[ ]:


#Fireplaces vs SalePrice
Fireplaces_pivot = train.pivot_table(index='Fireplaces', values='SalePrice', aggfunc=np.median)
Fireplaces_pivot.plot(kind='bar',color='blue')
plt.xlabel('Fireplaces')
plt.ylabel('SalePrice')
plt.show()


# In[ ]:


#TotRmsAbvGrd vs SalePrice
TotRmsAbvGrd_pivot = train.pivot_table(index='TotRmsAbvGrd', values='SalePrice', aggfunc=np.median)
TotRmsAbvGrd_pivot.plot(kind='bar',color='blue')
plt.xlabel('TotRmsAbvGrd')
plt.ylabel('SalePrice')
plt.show()


# In[ ]:


#KitchenAbvGr vs SalePrice
KitchenAbvGr_pivot = train.pivot_table(index='KitchenAbvGr', values='SalePrice', aggfunc=np.median)
KitchenAbvGr_pivot.plot(kind='bar',color='blue')
plt.xlabel('KitchenAbvGr')
plt.ylabel('SalePrice')
plt.show()


# # Detailed data analysis - Analysing continuous numeric variables

# In[ ]:


#GrLivArea Vs SalePrice
plt.scatter(x=train['GrLivArea'],y=targets)
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.show()


# In[ ]:


#GarageCars vs SalePrice
plt.scatter(x=train['GarageCars'],y=targets)
plt.xlabel('GarageCars')
plt.ylabel('SalePrice')
plt.show()


# In[ ]:


#GarageArea vs SalePrice
plt.scatter(x=train['GarageArea'],y=targets)
plt.xlabel('GarageArea')
plt.ylabel('SalePrice')
plt.show()


# In[ ]:


# Numeric variables vs SalePrice

sns.set_style('darkgrid')
plt.figure(figsize=(26,32))
plt.subplot(17,2,1)
plt.tight_layout()
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
plt.hist(combined['LotFrontage'], bins=30, edgecolor= 'black',color ='teal')
plt.title('LotFrontage')

plt.subplot(17,2,2)
plt.scatter(x=train.LotFrontage, y=train.SalePrice,edgecolor= 'black',color ='teal')
plt.title('LotFrontage vs SalePrice')

plt.subplot(17,2,3)
plt.hist(combined['MasVnrArea'], bins=30, edgecolor= 'black',color ='teal')
plt.title('MasVnrArea')

plt.subplot(17,2,4)
plt.scatter(x=train.MasVnrArea, y=train.SalePrice,edgecolor= 'black',color ='teal')
plt.title('MasVnrArea vs SalePrice')

plt.subplot(17,2,5)
plt.hist(combined['LotArea'], bins=30, edgecolor= 'black',color ='teal')
plt.title('LotArea')

plt.subplot(17,2,6)
plt.scatter(x=train.LotArea, y=train.SalePrice,edgecolor= 'black',color ='teal')
plt.title('LotArea vs SalePrice')

plt.subplot(17,2,7)
plt.hist(combined['BsmtFinSF1'], bins=30, edgecolor= 'black',color ='teal')
plt.title('BsmtFinSF1')

plt.subplot(17,2,8)
plt.scatter(x=train.BsmtFinSF1, y=train.SalePrice,edgecolor= 'black',color ='teal')
plt.title('BsmtFinSF1 vs SalePrice')

plt.subplot(17,2,9)
plt.hist(combined['BsmtFinSF2'], bins=30, edgecolor= 'black',color ='teal')
plt.title('BsmtFinSF2')

plt.subplot(17,2,10)
plt.scatter(x=train.BsmtFinSF2, y=train.SalePrice,edgecolor= 'black',color ='teal')
plt.title('BsmtFinSF2 vs SalePrice')

plt.subplot(17,2,11)
plt.hist(combined['BsmtUnfSF'], bins=30, edgecolor= 'black',color ='teal')
plt.title('BsmtUnfSF')

plt.subplot(17,2,12)
plt.scatter(x=train.BsmtUnfSF, y=train.SalePrice,edgecolor= 'black',color ='teal')
plt.title('BsmtUnfSF vs SalePrice')

plt.subplot(17,2,13)
plt.hist(combined['TotalBsmtSF'], bins=30, edgecolor= 'black',color ='teal')
plt.title('TotalBsmtSF')

plt.subplot(17,2,14)
plt.scatter(x=train.TotalBsmtSF, y=train.SalePrice,edgecolor= 'black',color ='teal')
plt.title('TotalBsmtSF vs SalePrice')

plt.subplot(17,2,15)
plt.hist(combined['1stFlrSF'], bins=30, edgecolor= 'black',color ='teal')
plt.title('1stFlrSF')

plt.subplot(17,2,16)
plt.scatter(x=train['1stFlrSF'], y=train.SalePrice,edgecolor= 'black',color ='teal')
plt.title('1stFlrSF vs SalePrice')

plt.subplot(17,2,17)
plt.hist(combined['2ndFlrSF'], bins=30, edgecolor= 'black',color ='teal')
plt.title('2ndFlrSF')

plt.subplot(17,2,18)
plt.scatter(x=train['2ndFlrSF'], y=train.SalePrice,edgecolor= 'black',color ='teal')
plt.title('2ndFlrSF vs SalePrice')

plt.subplot(17,2,19)
plt.hist(combined['LowQualFinSF'], bins=30, edgecolor= 'black',color ='teal')
plt.title('LowQualFinSF')

plt.subplot(17,2,20)
plt.scatter(x=train['LowQualFinSF'], y=train.SalePrice,edgecolor= 'black',color ='teal')
plt.title('LowQualFinSF vs SalePrice')

plt.subplot(17,2,21)
plt.hist(combined['WoodDeckSF'], bins=30, edgecolor= 'black',color ='teal')
plt.title('WoodDeckSF')

plt.subplot(17,2,22)
plt.scatter(x=train['WoodDeckSF'], y=train.SalePrice,edgecolor= 'black',color ='teal')
plt.title('WoodDeckSF vs SalePrice')

plt.subplot(17,2,23)
plt.hist(combined['OpenPorchSF'], bins=30, edgecolor= 'black',color ='teal')
plt.title('OpenPorchSF')

plt.subplot(17,2,24)
plt.scatter(x=train['OpenPorchSF'], y=train.SalePrice,edgecolor= 'black',color ='teal')
plt.title('OpenPorchSF vs SalePrice')

plt.subplot(17,2,25)
plt.hist(combined['EnclosedPorch'], bins=30, edgecolor= 'black',color ='teal')
plt.title('EnclosedPorch')

plt.subplot(17,2,26)
plt.scatter(x=train['EnclosedPorch'], y=train.SalePrice,edgecolor= 'black',color ='teal')
plt.title('EnclosedPorch vs SalePrice')

plt.subplot(17,2,27)
plt.hist(combined['3SsnPorch'], bins=30, edgecolor= 'black',color ='teal')
plt.title('3SsnPorch')

plt.subplot(17,2,28)
plt.scatter(x=train['3SsnPorch'], y=train.SalePrice,edgecolor= 'black',color ='teal')
plt.title('3SsnPorch vs SalePrice')

plt.subplot(17,2,29)
plt.hist(combined['ScreenPorch'], bins=30, edgecolor= 'black',color ='teal')
plt.title('ScreenPorch')

plt.subplot(17,2,30)
plt.scatter(x=train['ScreenPorch'], y=train.SalePrice,edgecolor= 'black',color ='teal')
plt.title('ScreenPorch vs SalePrice')

plt.subplot(17,2,31)
plt.hist(combined['PoolArea'], bins=30, edgecolor= 'black',color ='teal')
plt.title('PoolArea')

plt.subplot(17,2,32)
plt.scatter(x=train['PoolArea'], y=train.SalePrice,edgecolor= 'black',color ='teal')
plt.title('PoolArea vs SalePrice')

plt.subplot(17,2,33)
plt.hist(combined['MiscVal'], bins=30, edgecolor= 'black',color ='teal')
plt.title('MiscVal')

plt.subplot(17,2,34)
plt.scatter(x=train['MiscVal'], y=train.SalePrice,edgecolor= 'black',color ='teal')
plt.title('MiscVal vs SalePrice')

plt.show()


# # Observed Outliers
# ### LotFrontage > 300
# ### GarageArea > 1200
# ### GarageCars > 3.5
# ### LotFrontage > 300
# ### MasVnrArea > 1500
# ### BsmtFinSF2 > 5000
# ### 1stFlrSF > 4000

# # Detailed data analysis - Analysing categorical variables

# In[ ]:


#MSSubClass
sns.countplot(combined.MSSubClass)
plt.show()
sns.boxplot(x = 'MSSubClass', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by MSSubclass')
#Note correlation seems to be very less with Sale Price


# In[ ]:


#MSZoning
sns.countplot(combined.MSZoning)
plt.show()
sns.boxplot(x = 'MSZoning', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by MSZoning')


# In[ ]:


#Street
sns.countplot(combined.Street)
plt.show()
sns.violinplot(x = 'Street', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by Street')


# In[ ]:


#Utilities
sns.countplot(combined.Utilities) 
#Entire column can be deleted as it contains only one level


# In[ ]:


#Foundation
sns.countplot(combined.Foundation)
sns.violinplot(x = 'Foundation', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by Foundation')


# In[ ]:


#BsmtCond
sns.countplot(combined.BsmtCond)
plt.show()
sns.boxplot(x = 'BsmtCond', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by BsmtCond')


# In[ ]:


#BsmtQual
sns.countplot(combined.BsmtQual)
plt.show()
sns.boxplot(x = 'BsmtQual', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by BsmtCond')


# In[ ]:


#BsmtFinType1
sns.countplot(combined.BsmtFinType1)
plt.show()
sns.boxplot(x = 'BsmtFinType1', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by BsmtFinType1')


# In[ ]:


#BsmtFinType2
sns.countplot(combined.BsmtFinType2)
plt.show()
sns.boxplot(x = 'BsmtFinType2', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by BsmtFinType2')
#Majority is unfinished with clearly no impact on Sale Price


# In[ ]:


#LotShape 
sns.countplot(combined.LotShape)
plt.show()
sns.boxplot(x = 'LotShape', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by LotShape')


# In[ ]:


#Alley - Can conclude that paved alley leads to much higher prices than gravel
sns.countplot(combined.Alley)
plt.show()
sns.boxplot(x = 'Alley', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by Alley')


# In[ ]:


#ExterQual
sns.countplot(combined.ExterQual)
plt.show()
sns.boxplot(x = 'ExterQual', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by ExterQual')


# In[ ]:


#ExterCond
sns.countplot(combined.ExterCond)
plt.show()
sns.boxplot(x = 'ExterCond', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by ExterCond')
#Note most are TA, hardly any correlation in the data


# In[ ]:


#Heating
sns.countplot(combined.Heating)
plt.show()
sns.boxplot(x = 'Heating', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by Heating')
#Data dominated by one category(GasA)


# In[ ]:


#HeatingQC
sns.countplot(combined.HeatingQC)
plt.show()
sns.boxplot(x = 'HeatingQC', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by HeatingQC')


# In[ ]:


#CentralAir
sns.countplot(combined.CentralAir)
plt.show()
sns.boxplot(x = 'CentralAir', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by CentralAC')
#With CentralAC saleprices are high compared to NonAC


# In[ ]:


#FireplaceQu
sns.countplot(combined.FireplaceQu)
plt.show()
sns.boxplot(x = 'FireplaceQu', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by FireplaceQu')


# In[ ]:


#GarageType
sns.countplot(combined.GarageType)
plt.show()
sns.boxplot(x = 'GarageType', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by GarageType')


# In[ ]:


#GarageFinish
sns.countplot(combined.GarageFinish)
plt.show()
sns.boxplot(x = 'GarageFinish', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by GarageFinish')


# In[ ]:


#GarageQual
sns.countplot(combined.GarageQual)
plt.show()
sns.boxplot(x = 'GarageQual', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by GarageQual')


# In[ ]:


#Fence
sns.countplot(combined.Fence)
plt.show()
sns.boxplot(x = 'Fence', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by Fence')


# In[ ]:


#PavedDrive
sns.countplot(combined.PavedDrive)
plt.show()
sns.boxplot(x = 'PavedDrive', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by PavedDrive')


# In[ ]:


#LandSlope
sns.countplot(combined.LandSlope)
plt.show()
sns.boxplot(x = 'LandSlope', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by LandSlope')
#Not much corelation here!


# In[ ]:


#Condition1
sns.countplot(combined.Condition1)
plt.show()
sns.boxplot(x = 'Condition1', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by Condition1')


# In[ ]:


#Condition2
sns.countplot(combined.Condition2)
plt.show()
sns.boxplot(x = 'Condition2', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by Condition2')


# In[ ]:


#Functional
sns.countplot(combined.Functional)
plt.show()
sns.boxplot(x = 'Functional', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by Fuctional')


# In[ ]:


#KitchenQual
sns.countplot(combined.KitchenQual)
plt.show()
sns.boxplot(x = 'KitchenQual', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by KitchenQual')


# In[ ]:


#RoofMatl
sns.countplot(combined.RoofMatl)
plt.show()
sns.violinplot(x = 'RoofMatl', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by RoofMatl')
#Important indicator of SalePrice


# In[ ]:


#RoofStyle
sns.countplot(combined.RoofStyle)
plt.show()
sns.boxplot(x = 'RoofStyle', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by RoofStyle')


# In[ ]:


#Electrical
sns.countplot(combined.Electrical)
plt.show()
sns.boxplot(x = 'Electrical', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by Electrical')


# In[ ]:


#Exterior1st
sns.countplot(combined.Exterior1st)
plt.show()
sns.boxplot(x = 'Exterior1st', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by Exterior1st')


# In[ ]:


#Exterior2nd
sns.countplot(combined.Exterior2nd)
plt.show()
sns.boxplot(x = 'Exterior2nd', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by Exterior2nd')


# In[ ]:


#BldgType
sns.countplot(combined.BldgType) 
plt.show()
sns.boxplot(x = 'BldgType', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by BldgType')


# In[ ]:


#HouseStyle
sns.countplot(combined.HouseStyle)
plt.show()
sns.boxplot(x = 'HouseStyle', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by HouseStyle')


# In[ ]:


#LotConfig
sns.countplot(combined.LotConfig)
plt.show()
sns.boxplot(x = 'LotConfig', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by LotConfig')


# In[ ]:


#SaleType
sns.countplot(combined.SaleType)
plt.show()
sns.boxplot(x = 'SaleType', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by SaleType')


# In[ ]:


#SaleCondition
sns.countplot(combined.SaleCondition)
plt.show()
sns.boxplot(x = 'SaleCondition', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by SaleCondition')


# In[ ]:


#MasVnrType
sns.countplot(combined.MasVnrType)
plt.show()
sns.boxplot(x = 'MasVnrType', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by MasVnrType')


# In[ ]:


#MiscFeature
sns.countplot(combined.MiscFeature)
plt.show()
sns.boxplot(x = 'MiscFeature', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by MiscFeature')


# In[ ]:


#LandContour
sns.countplot(combined.LandContour)
plt.show()
sns.boxplot(x = 'LandContour', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by LandContour')


# In[ ]:


#Neighborhood
sns.countplot(combined.Neighborhood)
plt.xticks(rotation=45)
plt.show()
sns.stripplot(x = train.Neighborhood.values, y = train.SalePrice,
              order = np.sort(train.Neighborhood.unique()),
              jitter=0.1, alpha=0.5).set_title('Sale Price by Neighbourhood')
 
plt.xticks(rotation=45)


# # So if you are wondering why using various plots for EDA? - Obviously it is more pleasing when we present results visually rather than in raw format.
# 
# # Predictions coming up next.
