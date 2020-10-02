#!/usr/bin/env python
# coding: utf-8

# Load Package

# In[1]:



import numpy as np 
import pandas as pd 
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.model_selection import cross_val_score

import os
print(os.listdir("../input"))


# Load Data

# In[2]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# Data describe

# Here's a brief version of what you'll find in the data description file.
# 
# SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
# 
# MSSubClass: The building class
# 
# MSZoning: The general zoning classification
# 
# LotFrontage: Linear feet of street connected to property
# 
# LotArea: Lot size in square feet
# 
# Street: Type of road access
# 
# Alley: Type of alley access
# 
# LotShape: General shape of property
# 
# LandContour: Flatness of the property
# 
# Utilities: Type of utilities available
# 
# LotConfig: Lot configuration
# 
# LandSlope: Slope of property
# 
# Neighborhood: Physical locations within Ames city limits
# 
# Condition1: Proximity to main road or railroad
# 
# Condition2: Proximity to main road or railroad (if a second is present)
# 
# BldgType: Type of dwelling
# 
# HouseStyle: Style of dwelling
# 
# OverallQual: Overall material and finish quality
# 
# OverallCond: Overall condition rating
# 
# YearBuilt: Original construction date
# 
# YearRemodAdd: Remodel date
# 
# RoofStyle: Type of roof
# 
# RoofMatl: Roof material
# 
# Exterior1st: Exterior covering on house
# 
# Exterior2nd: Exterior covering on house (if more than one material)
# 
# MasVnrType: Masonry veneer type
# 
# MasVnrArea: Masonry veneer area in square feet
# 
# ExterQual: Exterior material quality
# 
# ExterCond: Present condition of the material on the exterior
# 
# Foundation: Type of foundation
# 
# BsmtQual: Height of the basement
# 
# BsmtCond: General condition of the basement
# 
# BsmtExposure: Walkout or garden level basement walls
# 
# BsmtFinType1: Quality of basement finished area
# 
# BsmtFinSF1: Type 1 finished square feet
# 
# BsmtFinType2: Quality of second finished area (if present)
# 
# BsmtFinSF2: Type 2 finished square feet
# 
# BsmtUnfSF: Unfinished square feet of basement area
# 
# TotalBsmtSF: Total square feet of basement area
# 
# Heating: Type of heating
# 
# HeatingQC: Heating quality and condition
# 
# CentralAir: Central air conditioning
# 
# Electrical: Electrical system
# 
# 1stFlrSF: First Floor square feet
# 
# 2ndFlrSF: Second floor square feet
# 
# LowQualFinSF: Low quality finished square feet (all floors)
# 
# GrLivArea: Above grade (ground) living area square feet
# 
# BsmtFullBath: Basement full bathrooms
# 
# BsmtHalfBath: Basement half bathrooms
# 
# FullBath: Full bathrooms above grade
# 
# HalfBath: Half baths above grade
# 
# Bedroom: Number of bedrooms above basement level
# 
# Kitchen: Number of kitchens
# 
# KitchenQual: Kitchen quality
# 
# TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
# 
# Functional: Home functionality rating
# 
# Fireplaces: Number of fireplaces
# 
# FireplaceQu: Fireplace quality
# 
# 
# GarageType: Garage location
# 
# GarageYrBlt: Year garage was built
# 
# GarageFinish: Interior finish of the garage
# 
# GarageCars: Size of garage in car capacity
# 
# GarageArea: Size of garage in square feet
# 
# GarageQual: Garage quality
# 
# GarageCond: Garage condition
# 
# PavedDrive: Paved driveway
# 
# WoodDeckSF: Wood deck area in square feet
# 
# OpenPorchSF: Open porch area in square feet
# 
# EnclosedPorch: Enclosed porch area in square feet
# 
# 3SsnPorch: Three season porch area in square feet
# 
# ScreenPorch: Screen porch area in square feet
# 
# PoolArea: Pool area in square feet
# 
# PoolQC: Pool quality
# 
# Fence: Fence quality
# 
# MiscFeature: Miscellaneous feature not covered in other categories
# 
# MiscVal: $Value of miscellaneous feature
# 
# MoSold: Month Sold
# 
# YrSold: Year Sold
# 
# SaleType: Type of sale
# 
# SaleCondition: Condition of sale
# 

# In[3]:


print(train.shape)
print(test.shape)
print("train has {} data | test has {} data".format(train.shape[0], test.shape[0]))


# In[4]:


train.head()


# In[91]:


test.head()


# Explore Data

# In[6]:


print("train has {} columns".format(len(train.columns)))
print("test has {} columns".format(len(test.columns)))
print("Target data is SalePrice")


# Unique Data

# In[7]:


# check unique data in train
for i in train.columns:
    print("{} has {} unique data".format(i, len(train[i].unique())))


# Missing Data

# In[8]:


#check missing data in train
missing_train={}
for i in train.columns:
    x = (len(train[train[i].isnull()]) / (train.shape[0])) *100
    if x !=0:
        print("{0} has {1:.2f}% missing data".format(i, (len(train[train[i].isnull()]) / train.shape[0]) *100)) 
        missing_train[i] = (len(train[train[i].isnull()]) / (train.shape[0]) *100)


# In[9]:


#check missing data in train
missing_test ={}
for i in test.columns:
    x = (len(test[test[i].isnull()]) / (test.shape[0])) *100
    if x !=0:
        print("{0} has {1:.2f}% missing data".format(i, (len(test[test[i].isnull()]) / test.shape[0]) *100))
        missing_test[i] =(len(test[test[i].isnull()]) / (test.shape[0]) *100)


# Define Data

# In[10]:


num = []
cat = []
for i in train.columns:
    if train[i].dtype == object:
        cat.append(i)
    else:
        num.append(i)


# In[11]:


num
num.remove("Id")
num.remove("SalePrice")


# In[12]:


cat[0:10]


# In[13]:


train[num].head()


# Preprocessing

# Address Missing Data

# it is not a missing data. I assume that there are a lot of houses do not have Alley

# In[14]:


train.loc[train["Alley"].isnull(),"Alley"] = "None"
train["Alley"].value_counts()


# In[15]:


test.loc[test["Alley"].isnull(),"Alley"] = "None"
test["Alley"].value_counts()


# BsmtCond

# In[16]:


train["BsmtCond"].value_counts()
test["BsmtCond"].value_counts()


# In[17]:


train.loc[train["BsmtCond"].isnull(), "BsmtCond"] = 'None'
test.loc[test["BsmtCond"].isnull(), "BsmtCond"] = 'None'


# BsmExposure

# In[18]:


train["BsmtExposure"].value_counts()


# In[19]:


test["BsmtFinType1"].value_counts()


# In[20]:


train.loc[train["BsmtFinType1"].isnull(), "BsmtFinType1"] = 'None'
test.loc[test["BsmtFinType1"].isnull(), "BsmtFinType1"] = 'None'


# BsmtFinType2

# In[21]:


train["BsmtFinType2"].value_counts()


# In[22]:


test["BsmtFinType2"].value_counts()


# In[23]:


train.loc[train["BsmtFinType2"].isnull(), "BsmtFinType2"] = 'None'
test.loc[test["BsmtFinType2"].isnull(), "BsmtFinType2"] = 'None'


# BsmtQual

# In[24]:


train["BsmtQual"].value_counts()


# In[25]:


test["BsmtQual"].value_counts()


# In[26]:


train.loc[train["BsmtQual"].isnull(), "BsmtQual"] = 'None'
train.loc[train["BsmtQual"].isnull(), "BsmtQual"] = 'None'


# Electrical

# In[27]:


train["Electrical"].value_counts()


# In[28]:


test["Electrical"].value_counts()


# In[29]:


train.loc[train["Electrical"].isnull(), "Electrical"] = train["Electrical"].mode()[0]
test.loc[test["Electrical"].isnull(), "Electrical"] = test["Electrical"].mode()[0]


# Fence

# In[30]:


train["Fence"].value_counts()


# In[31]:


test["Fence"].value_counts()


# In[32]:


train.loc[train["Fence"].isnull(), "Fence"] = "None"
test.loc[test["Fence"].isnull(), "Fence"] = "None"


# FireplaceQu

# In[33]:


train["FireplaceQu"].value_counts()


# In[34]:


test["FireplaceQu"].value_counts()


# In[35]:


train.loc[train["FireplaceQu"].isnull(), "FireplaceQu"] = "None"
test.loc[test["FireplaceQu"].isnull(), "FireplaceQu"] = "None"


# GarageCond, Finish, Qual, Type, YrBlt

# In[36]:


train["GarageCond"].value_counts()


# In[37]:


train["GarageFinish"].value_counts()


# In[38]:


train["GarageQual"].value_counts()


# In[39]:


train["GarageType"].value_counts()


# In[40]:


train.loc[train["GarageCond"].isnull(), "GarageCond"] = "None"
train.loc[train["GarageFinish"].isnull(), "GarageFinish"] = "None"
train.loc[train["GarageQual"].isnull(), "GarageQual"] = "None"
train.loc[train["GarageType"].isnull(), "GarageType"] = "None"
train.loc[train["GarageYrBlt"].isnull(), "GarageYrBlt"] = 0


# In[41]:


test["GarageCond"].value_counts()


# In[42]:


test.loc[test["GarageCond"].isnull(), "GarageCond"] = "None"
test.loc[test["GarageFinish"].isnull(), "GarageFinish"] = "None"
test.loc[test["GarageQual"].isnull(), "GarageQual"] = "None"
test.loc[test["GarageType"].isnull(), "GarageType"] = "None"
test.loc[test["GarageYrBlt"].isnull(), "GarageYrBlt"] = 0


# LotFrontage

# There are non-zero in LotArea, LotFrontage must be replaced with mean  (LotArea = LotFrontage* Depth)

# In[43]:


train.loc[train["LotFrontage"]==0, "LotFrontage"] = round(train["LotFrontage"].mean())
test.loc[test["LotFrontage"]== 0, "LotFrontage"] = round(test["LotFrontage"].mean())


# MasVnrArea Type

# In[44]:


train.loc[train["MasVnrArea"].isnull(),"MasVnrArea"] = 0
train.loc[train["MasVnrType"].isnull(),"MasVnrType"] ="None"
test.loc[test["MasVnrArea"].isnull(),"MasVnrArea"] = 0
test.loc[test["MasVnrType"].isnull(),"MasVnrType"] ="None"


# MiscFeature

# Miscellaneous feature not covered in other categories

# In[45]:


train.loc[train["MiscFeature"].isnull(),"MiscFeature"] ="None"
test.loc[test["MiscFeature"].isnull(),"MiscFeature"] ="None"


# PoolQc

# In[46]:


train.loc[train["PoolQC"].isnull(),"PoolQC"] ="None"
test.loc[test["PoolQC"].isnull(),"PoolQC"] ="None"


# Address missing data that exist only in Test data

# MsZoning

# In[47]:


test["MSZoning"].value_counts()


# In[48]:


test.loc[test["MSZoning"].isnull(),"MSZoning"] =test["MSZoning"].mode()[0]


# Utilities

# In[49]:


test["Utilities"].value_counts()


# In[50]:


test.loc[test["Utilities"].isnull(),"Utilities"] ="None"


# Exterior1st ,2nd

# In[51]:


test.loc[test["Exterior1st"].isnull(),"Exterior1st"] ="None"
test.loc[test["Exterior2nd"].isnull(),"Exterior2nd"] ="None"


# BsmtFinSF1 ,SF2,UnfSF, Total 

# In[52]:


test.loc[test["BsmtFinSF1"].isnull(),"BsmtFinSF1"] = 0
test.loc[test["BsmtFinSF2"].isnull(),"BsmtFinSF2"] = 0
test.loc[test["BsmtUnfSF"].isnull(),"BsmtUnfSF"] = 0
test.loc[test["TotalBsmtSF"].isnull(),"TotalBsmtSF"] = 0


# BsmtFullBath, BsmtHalfBath 

# In[53]:


test.loc[test["BsmtFullBath"].isnull(),"BsmtFullBath"] = 0
test.loc[test["BsmtHalfBath"].isnull(),"BsmtHalfBath"] = 0


# GarageCars,GarageArea 

# In[54]:


test.loc[test["GarageCars"].isnull(),"GarageCars"] = 0
test.loc[test["GarageArea"].isnull(),"GarageArea"] = 0


# KitchenQual 

# In[55]:


test.loc[test["KitchenQual"].isnull(),"KitchenQual"] ="None"


# Functional 

# In[56]:


test.loc[test["Functional"].isnull(),'Functional'] = test["Functional"].mode()[0]


# SaleType

# In[57]:


test.loc[test["SaleType"].isnull(),'SaleType'] = test["SaleType"].mode()[0]


# Encode Feature

# In[58]:


cat_train = train[cat]


# In[59]:


cat_train.shape


# In[60]:


cat_test = test[cat]
cat_all = pd.concat([cat_train, cat_test], axis= 0)
# one_hot_encoding
cat_dum_all = pd.get_dummies(cat_all)
cat_dum_all.shape


# In[61]:


cat_dum_train = cat_dum_all[0:1460]
cat_dum_train.shape


# In[62]:


cat_dum_test = cat_dum_all[1460:2921]
cat_dum_test.shape


# In[63]:


train.drop(cat_train,inplace=True, axis =1)
train_dum = pd.concat([train,cat_dum_train], axis=1)


# In[64]:


train_dum.head()


# In[65]:


test.drop(cat_test,inplace=True, axis =1)
test_dum = pd.concat([test,cat_dum_test], axis=1)
test_dum.head()


# Feature Engineering

# Make Feauture polunomial type

# In[66]:


figure, ((ax1,ax2,ax3,ax4),(ax5,ax6,ax7,ax8),(ax9,ax10,ax11,ax12),
        (ax13,ax14,ax15,ax16),(ax17,ax18,ax19,ax20),(ax21,ax22,ax23,ax24),
        (ax25,ax26,ax27,ax28),(ax29,ax30,ax31,ax32),(ax33,ax34,ax35,ax36)) =plt.subplots(nrows=9, ncols =4)
figure.set_size_inches(20,20)
sns.pointplot(train["MSSubClass"],train["SalePrice"],ax=ax1)
sns.pointplot(train["LotFrontage"],train["SalePrice"],ax=ax2)
sns.pointplot(train["LotArea"],train["SalePrice"],ax=ax3)
sns.pointplot(train["OverallQual"],train["SalePrice"],ax=ax4)
sns.pointplot(train["OverallCond"],train["SalePrice"],ax=ax5)
sns.pointplot(train["YearBuilt"],train["SalePrice"],ax=ax6)
sns.pointplot(train["YearRemodAdd"],train["SalePrice"],ax=ax7)
sns.pointplot(train["MasVnrArea"],train["SalePrice"],ax=ax8)
sns.pointplot(train["BsmtFinSF1"],train["SalePrice"],ax=ax9)
sns.pointplot(train["BsmtFinSF2"],train["SalePrice"],ax=ax10)
sns.pointplot(train["BsmtUnfSF"],train["SalePrice"],ax=ax11)
sns.pointplot(train["TotalBsmtSF"],train["SalePrice"],ax=ax12)
sns.pointplot(train["1stFlrSF"],train["SalePrice"],ax=ax13)
sns.pointplot(train["2ndFlrSF"],train["SalePrice"],ax=ax14)
sns.pointplot(train["LowQualFinSF"],train["SalePrice"],ax=ax15)
sns.pointplot(train["GrLivArea"],train["SalePrice"],ax=ax16)
sns.pointplot(train["BsmtFullBath"],train["SalePrice"],ax=ax17)
sns.pointplot(train["BsmtHalfBath"],train["SalePrice"],ax=ax18)
sns.pointplot(train["FullBath"],train["SalePrice"],ax=ax19)
sns.pointplot(train["HalfBath"],train["SalePrice"],ax=ax20)
sns.pointplot(train["BedroomAbvGr"],train["SalePrice"],ax=ax21)
sns.pointplot(train["KitchenAbvGr"],train["SalePrice"],ax=ax22)
sns.pointplot(train["TotRmsAbvGrd"],train["SalePrice"],ax=ax23)
sns.pointplot(train["Fireplaces"],train["SalePrice"],ax=ax24)
sns.pointplot(train["GarageYrBlt"],train["SalePrice"],ax=ax25)
sns.pointplot(train["GarageCars"],train["SalePrice"],ax=ax26)
sns.pointplot(train["GarageArea"],train["SalePrice"],ax=ax27)
sns.pointplot(train["WoodDeckSF"],train["SalePrice"],ax=ax28)
sns.pointplot(train["OpenPorchSF"],train["SalePrice"],ax=ax29)
sns.pointplot(train["EnclosedPorch"],train["SalePrice"],ax=ax30)
sns.pointplot(train["3SsnPorch"],train["SalePrice"],ax=ax31)
sns.pointplot(train["ScreenPorch"],train["SalePrice"],ax=ax32)
sns.pointplot(train["PoolArea"],train["SalePrice"],ax=ax33)
sns.pointplot(train["MiscVal"],train["SalePrice"],ax=ax34)
sns.pointplot(train["MoSold"],train["SalePrice"],ax=ax35)
sns.pointplot(train["YrSold"],train["SalePrice"],ax=ax36)


# In[67]:


#train
train_dum["OverallQual-s2"] = train_dum["OverallQual"] ** 2
train_dum["OverallQual-s3"] = train_dum["OverallQual"] ** 3
train_dum["OverallQual-Sq"] = np.sqrt(train_dum["OverallQual"])


train_dum["OverallCond-s2"] = train_dum["OverallCond"] ** 2
train_dum["OverallCond-s3"] = train_dum["OverallCond"] ** 3
train_dum["OverallCond-Sq"] = np.sqrt(train_dum["OverallCond"])

train_dum["YearRemodAdd-s2"] = train_dum["YearRemodAdd"] ** 2
train_dum["YearRemodAdd-s3"] = train_dum["YearRemodAdd"] ** 3
train_dum["YearRemodAdd-Sq"] = np.sqrt(train_dum["YearRemodAdd"])

train_dum["FullBath-s2"] = train_dum["FullBath"] ** 2
train_dum["FullBath-s3"] = train_dum["FullBath"] ** 3
train_dum["FullBath-Sq"] = np.sqrt(train_dum["FullBath"])

train_dum["TotRmsAbvGrd-s2"] = train_dum["TotRmsAbvGrd"] ** 2
train_dum["TotRmsAbvGrd-s3"] = train_dum["TotRmsAbvGrd"] ** 3
train_dum["TotRmsAbvGrd-Sq"] = np.sqrt(train_dum["TotRmsAbvGrd"])


train_dum["Fireplaces-s2"] = train_dum["Fireplaces"] ** 2
train_dum["Fireplaces-s3"] = train_dum["Fireplaces"] ** 3
train_dum["Fireplaces-Sq"] = np.sqrt(train_dum["Fireplaces"])


# In[68]:


#train
test_dum["OverallQual-s2"] = test_dum["OverallQual"] ** 2
test_dum["OverallQual-s3"] = test_dum["OverallQual"] ** 3
test_dum["OverallQual-Sq"] = np.sqrt(test_dum["OverallQual"])


test_dum["OverallCond-s2"] = test_dum["OverallCond"] ** 2
test_dum["OverallCond-s3"] = test_dum["OverallCond"] ** 3
test_dum["OverallCond-Sq"] = np.sqrt(test_dum["OverallCond"])

test_dum["YearRemodAdd-s2"] = test_dum["YearRemodAdd"] ** 2
test_dum["YearRemodAdd-s3"] = test_dum["YearRemodAdd"] ** 3
test_dum["YearRemodAdd-Sq"] = np.sqrt(test_dum["YearRemodAdd"])

test_dum["FullBath-s2"] = test_dum["FullBath"] ** 2
test_dum["FullBath-s3"] = test_dum["FullBath"] ** 3
test_dum["FullBath-Sq"] = np.sqrt(test_dum["FullBath"])

test_dum["TotRmsAbvGrd-s2"] = test_dum["TotRmsAbvGrd"] ** 2
test_dum["TotRmsAbvGrd-s3"] = test_dum["TotRmsAbvGrd"] ** 3
test_dum["TotRmsAbvGrd-Sq"] = np.sqrt(test_dum["TotRmsAbvGrd"])


test_dum["Fireplaces-s2"] = test_dum["Fireplaces"] ** 2
test_dum["Fireplaces-s3"] = test_dum["Fireplaces"] ** 3
test_dum["Fireplaces-Sq"] = np.sqrt(test_dum["Fireplaces"])


# In[69]:


add = ["OverallQual-s2","OverallQual-s3","OverallQual-Sq","OverallCond-s2","OverallCond-s3","OverallCond-Sq",
 "YearRemodAdd-s2","YearRemodAdd-s3","YearRemodAdd-Sq","FullBath-s2","FullBath-s3","FullBath-Sq",
 "TotRmsAbvGrd-s2","TotRmsAbvGrd-s3","TotRmsAbvGrd-Sq","Fireplaces-s2","Fireplaces-s3","Fireplaces-Sq"]


# In[70]:


num = num + add


# In[71]:


train_dum.loc[train_dum["LotFrontage"].isnull(),"LotFrontage"] = train_dum["LotFrontage"].mean()


# In[72]:


test_dum.loc[test_dum["LotFrontage"].isnull(),"LotFrontage"] = test_dum["LotFrontage"].mean()


# Feature Selection

# In[73]:


scaler = MinMaxScaler()
train_dum[num] = scaler.fit_transform(train_dum[num])


# In[74]:


test_dum.loc[test_dum["BsmtFinSF1"].isnull(),"BsmtFinSF1"] =0
test_dum.loc[test_dum["BsmtFinSF2"].isnull(),"BsmtFinSF2"] = 0
test_dum.loc[test_dum["BsmtUnfSF"].isnull(),"BsmtUnfSF"] = 0
test_dum.loc[test_dum["TotalBsmtSF"].isnull(),"TotalBsmtSF"] =0


# In[75]:


test_dum[num] = scaler.fit_transform(test_dum[num])


# In[76]:


x_train = train_dum.drop("SalePrice",axis=1)
x_train.set_index("Id",inplace=True)
x_test = test_dum.set_index("Id")


# In[77]:


y_train = train["SalePrice"]


# In[78]:


y_train = np.log(y_train)


# selected by feature importance

# In[90]:


select_feat = ['LotArea',
 'GrLivArea',
 'BsmtUnfSF',
 'TotalBsmtSF',
 '1stFlrSF',
 'GarageArea',
 'BsmtFinSF1',
 'MasVnrArea',
 'WoodDeckSF',
 '2ndFlrSF',
 'OpenPorchSF',
 'YearRemodAdd',
 'YearBuilt',
 'LotFrontage',
 'GarageYrBlt',
 'MoSold',
 'YrSold',
 'OverallQual',
 'TotRmsAbvGrd',
 'MSSubClass',
 'OverallCond',
 'FireplaceQu_TA',
 'HeatingQC_TA',
 'BedroomAbvGr',
 'Fireplaces',
 'LotShape_IR1',
 'BsmtExposure_No',
 'GarageFinish_Fin',
 'GarageFinish_RFn',
 'EnclosedPorch',
 'GarageFinish_Unf',
 'MasVnrType_BrkFace',
 'BsmtQual_Gd',
 'KitchenQual_TA',
 'LotShape_Reg',
 'KitchenQual_Gd',
 'Foundation_CBlock',
 'FullBath',
 'BsmtFinType1_GLQ',
 'HeatingQC_Gd',
 'BsmtFullBath',
 'BsmtFinType1_ALQ',
 'LotConfig_Inside',
 'HeatingQC_Ex',
 'HalfBath',
 'GarageType_Detchd',
 'Neighborhood_CollgCr',
 'FireplaceQu_Gd',
 'LotConfig_Corner',
 'BsmtExposure_Av']


# Train

# In[93]:


x_train = x_train[select_feat]
x_test = x_test[select_feat]


# In[94]:


model = xgb.XGBRegressor(nthread = 4,learning_rate =0.038264, reg_alpha =0.203944,
                        n_estimators=276)


# In[95]:


model.fit(x_train,y_train)


# In[96]:


predictions = model.predict(x_test)


# In[97]:


predictions = model.predict(x_test)


# In[98]:


predictions = np.exp(predictions)


# Train

# In[ ]:


x_train = x_train[select_feat]
x_test = x_test[select_feat]


# In[ ]:


model = xgb.XGBRegressor(nthread = 4,learning_rate =0.038264, reg_alpha =0.203944,
                        n_estimators=276)


# In[ ]:


model.fit(x_train,y_train)


# In[ ]:


predictions = model.predict(x_test)
predictions = np.exp(predictions)
predictions[0:10]


# In[99]:


predictions[0:100]


# submit

# In[101]:


submit = pd.read_csv("../input/sample_submission.csv")
submit["SalePrice"] = predictions
submit.set_index("Id",inplace=True)
submit.to_csv("submit.csv")


# Kaggle Score:0.15292

# In[ ]:




