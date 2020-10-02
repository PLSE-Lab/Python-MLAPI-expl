#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale


# In[ ]:


df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
df_test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")


# # **Data quality report**
# 

# In[ ]:


df.info()
df.describe()


# In[ ]:


sns.set_context('talk')
sns.set_color_codes()


# In[ ]:


plt1 = sns.barplot(x="MSSubClass", y="SalePrice", data=df[['MSSubClass', 'SalePrice']])


# In[ ]:


plt2 = sns.distplot(df['MSSubClass'].dropna())


# MSSubClass resume:
#   *  Distribution is skewed left, but the barplot shows that there is no 1 to 1 correlation 
#   *  The MSSubClass column has a clear numeric data, but it should be organized as bins
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.barplot(x="MSZoning", y="SalePrice", data=df[['MSZoning', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['MSZoning'].dropna())


# MSZoning resume:
#   *  Distribution is normal with some anomalies, but the barplot shows that there is no 1 to 1 correlation 
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with label encoding

# In[ ]:


plt1 = sns.lineplot(x="LotFrontage", y="SalePrice", data=df[['LotFrontage', 'SalePrice']])


# In[ ]:


plt2 = sns.distplot(df['LotFrontage'].dropna())


# LotFrontage resume:
#   *  Distribution is normal with some anomalies, but the lineplot shows that there is no 1 to 1 correlation 
#   *  Numeric data
#   
# Verdict:
#   *  Use this column in ML with "bins encoding"

# In[ ]:


plt1 = sns.lineplot(x="LotArea", y="SalePrice", data=df[['LotArea', 'SalePrice']])


# In[ ]:


plt2 = sns.distplot(df['LotArea'].dropna())


# LotFrontage resume:
#   *  Distribution is normal, but the lineplot shows that there is no 1 to 1 correlation 
#   *  Numeric data
#   
# Verdict:
#   *  Use this column in ML with "bins encoding"

# In[ ]:


plt1 = sns.barplot(x="Street", y="SalePrice", data=df[['Street', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['Street'])


# Street resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation 
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.barplot(x="Alley", y="SalePrice", data=df[['Alley', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['Alley'])


# Alley resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation 
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.barplot(x="LotShape", y="SalePrice", data=df[['LotShape', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['LotShape'])


# LotShape resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation 
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.barplot(x="LandContour", y="SalePrice", data=df[['LandContour', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['LandContour'])


# LandContour resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation 
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.barplot(x="Utilities", y="SalePrice", data=df[['Utilities', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['Utilities'])


# Utilities resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation 
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.barplot(x="LotConfig", y="SalePrice", data=df[['LotConfig', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['LotConfig'])


# LotConfig resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation 
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.barplot(x="LandSlope", y="SalePrice", data=df[['LandSlope', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['LandSlope'])


# LandSlope resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation 
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.barplot(x="Neighborhood", y="SalePrice", data=df[['Neighborhood', 'SalePrice']])
x = plt1.set_xticklabels(plt1.get_xticklabels(), rotation=40, ha="right", fontsize=7)


# In[ ]:


plt2 = sns.countplot(df['Neighborhood'])
x = plt2.set_xticklabels(plt2.get_xticklabels(), rotation=40, ha="right", fontsize=7)


# Neighborhood resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation 
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.barplot(x="Condition1", y="SalePrice", data=df[['Condition1', 'SalePrice']])
x = plt1.set_xticklabels(plt1.get_xticklabels(), rotation=40, ha="right")


# In[ ]:


plt2 = sns.countplot(df['Condition1'])
x = plt2.set_xticklabels(plt2.get_xticklabels(), rotation=40, ha="right")


# Condition1 resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation 
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.barplot(x="Condition2", y="SalePrice", data=df[['Condition2', 'SalePrice']])
x = plt1.set_xticklabels(plt1.get_xticklabels(), rotation=40, ha="right")


# In[ ]:


plt2 = sns.countplot(df['Condition2'])
x = plt2.set_xticklabels(plt2.get_xticklabels(), rotation=40, ha="right")


# Condition2 resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation 
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.barplot(x="BldgType", y="SalePrice", data=df[['BldgType', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['BldgType'])


# BldgType resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation 
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.barplot(x="HouseStyle", y="SalePrice", data=df[['HouseStyle', 'SalePrice']])
x = plt1.set_xticklabels(plt1.get_xticklabels(), rotation=40, ha="right")


# In[ ]:


plt2 = sns.countplot(df['HouseStyle'])
x = plt2.set_xticklabels(plt2.get_xticklabels(), rotation=40, ha="right")


# HouseStyle resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation 
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.barplot(x="OverallQual", y="SalePrice", data=df[['OverallQual', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['OverallQual'])


# OverallQual resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation 
#   *  Numeric data
#   
# Verdict:
#   *  Use this column in ML

# In[ ]:


plt1 = sns.barplot(x="OverallCond", y="SalePrice", data=df[['OverallCond', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['OverallCond'])


# OverallCond resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation 
#   *  Numeric data
#   
# Verdict:
#   *  Use this column in ML

# In[ ]:


plt1 = sns.lineplot(x="YearBuilt", y="SalePrice", data=df[['YearBuilt', 'SalePrice']])


# YearBuilt resume:
#   *  The lineplot shows that there is no 1 to 1 correlation 
#   *  Numeric data
#   
# Verdict:
#   *  Use this column in ML with "bins encoding"

# In[ ]:


plt1 = sns.lineplot(x="YearRemodAdd", y="SalePrice", data=df[['YearRemodAdd', 'SalePrice']])


# YearRemodAdd resume:
#   *  The lineplot shows that there is no 1 to 1 correlation 
#   *  Numeric data
#   
# Verdict:
#   *  Use this column in ML with "bins encoding"

# In[ ]:


plt1 = sns.barplot(x="RoofStyle", y="SalePrice", data=df[['RoofStyle', 'SalePrice']])
x = plt1.set_xticklabels(plt1.get_xticklabels(),rotation=40, ha="right")


# In[ ]:


plt2 = sns.countplot(df['RoofStyle'])
x = plt2.set_xticklabels(plt2.get_xticklabels(),rotation=40, ha="right")


# RoofStyle resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation 
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.barplot(x="RoofMatl", y="SalePrice", data=df[['RoofMatl', 'SalePrice']])
x = plt1.set_xticklabels(plt1.get_xticklabels(),rotation=40, ha="right")


# In[ ]:


plt2 = sns.countplot(df['RoofMatl'])
x = plt2.set_xticklabels(plt2.get_xticklabels(),rotation=40, ha="right")


# RoofMatl resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation 
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.barplot(x="Exterior1st", y="SalePrice", data=df[['Exterior1st', 'SalePrice']])
x = plt1.set_xticklabels(plt1.get_xticklabels(),rotation=55, ha="right")


# In[ ]:


plt2 = sns.countplot(df['Exterior1st'])
x = plt2.set_xticklabels(plt2.get_xticklabels(),rotation=55, ha="right")


# Exterior1st resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation 
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.barplot(x="Exterior2nd", y="SalePrice", data=df[['Exterior2nd', 'SalePrice']])
x = plt1.set_xticklabels(plt1.get_xticklabels(),rotation=55, ha="right")


# In[ ]:


plt2 = sns.countplot(df['Exterior2nd'])
x = plt2.set_xticklabels(plt2.get_xticklabels(),rotation=55, ha="right")


# Exterior2nd resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation 
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.barplot(x="MasVnrType", y="SalePrice", data=df[['MasVnrType', 'SalePrice']])
x = plt1.set_xticklabels(plt1.get_xticklabels(),rotation=40, ha="right")


# In[ ]:


plt2 = sns.countplot(df['MasVnrType'])
x = plt2.set_xticklabels(plt2.get_xticklabels(),rotation=40, ha="right")


# MasVnrType resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation 
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.lineplot(x="MasVnrArea", y="SalePrice", data=df[['MasVnrArea', 'SalePrice']])


# In[ ]:


plt2 = sns.distplot(df['MasVnrArea'].dropna())


# MasVnrArea resume:
#   *  The lineplot and the distplot show that there is no 1 to 1 correlation 
#   *  Numeric data
#   
# Verdict:
#   *  Use this column in ML with "bins encoding"

# In[ ]:


plt1 = sns.barplot(x="ExterQual", y="SalePrice", data=df[['ExterQual', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['ExterQual'])


# ExterQual resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation 
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.barplot(x="ExterCond", y="SalePrice", data=df[['ExterCond', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['ExterCond'])


# ExterCond resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation 
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.barplot(x="Foundation", y="SalePrice", data=df[['Foundation', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['Foundation'])


# Foundation resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation 
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.barplot(x="BsmtQual", y="SalePrice", data=df[['BsmtQual', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['BsmtQual'])


# BsmtQual resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation 
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# **Data quality plan**
# 

# In[ ]:


plt1 = sns.barplot(x="BsmtCond", y="SalePrice", data=df[['BsmtCond', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['BsmtCond'])


# BsmtCond resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation 
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.barplot(x="BsmtExposure", y="SalePrice", data=df[['BsmtExposure', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['BsmtExposure'])


# BsmtExposure resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation 
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.barplot(x="BsmtFinType1", y="SalePrice", data=df[['BsmtFinType1', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['BsmtFinType1'])


# BsmtFinType1 resume:
#   *  We'll check the correlation later more precisely
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.lineplot(x="BsmtFinSF1", y="SalePrice", data=df[['BsmtFinSF1', 'SalePrice']])


# In[ ]:


plt2 = sns.distplot(df['BsmtFinSF1'])


# BsmtFinSF1 resume:
#   *  The lineplot and the distplot show that there is no 1 to 1 correlation 
#   *  Numeric data
#   
# Verdict:
#   *  Use this column in ML with "bins encoding"

# In[ ]:


plt1 = sns.barplot(x="BsmtFinType2", y="SalePrice", data=df[['BsmtFinType2', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['BsmtFinType2'])


# BsmtFinType2 resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.lineplot(x="BsmtFinSF2", y="SalePrice", data=df[['BsmtFinSF2', 'SalePrice']])


# In[ ]:


plt2 = sns.distplot(df['BsmtFinSF2'])


# BsmtFinSF1 resume:
#   *  The lineplot and the distplot show that there is no 1 to 1 correlation 
#   *  Numeric data
#   
# Verdict:
#   *  Use this column in ML with "bins encoding"

# In[ ]:


plt1 = sns.lineplot(x="BsmtUnfSF", y="SalePrice", data=df[['BsmtUnfSF', 'SalePrice']])


# In[ ]:


plt2 = sns.distplot(df['BsmtUnfSF'])


# BsmtUnfSF resume:
#   *  The lineplot and the distplot show that there is no 1 to 1 correlation 
#   *  Numeric data
#   
# Verdict:
#   *  Use this column in ML with "bins encoding"

# In[ ]:


plt1 = sns.lineplot(x="TotalBsmtSF", y="SalePrice", data=df[['TotalBsmtSF', 'SalePrice']])


# In[ ]:


plt2 = sns.distplot(df['TotalBsmtSF'])


# TotalBsmtSF resume:
#   *  The lineplot and the distplot show that there is no 1 to 1 correlation 
#   *  Numeric data
#   *  Normal distribution
#   
# Verdict:
#   *  Use this column in ML with "bins encoding"

# In[ ]:


plt1 = sns.barplot(x="Heating", y="SalePrice", data=df[['Heating', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['Heating'])


# Heating resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.barplot(x="HeatingQC", y="SalePrice", data=df[['HeatingQC', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['HeatingQC'])


# HeatingQC resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.barplot(x="CentralAir", y="SalePrice", data=df[['CentralAir', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['CentralAir'])


# CentralAir resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.barplot(x="Electrical", y="SalePrice", data=df[['Electrical', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['Electrical'])


# Electrical resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.lineplot(x="1stFlrSF", y="SalePrice", data=df[['1stFlrSF', 'SalePrice']])


# In[ ]:


plt2 = sns.distplot(df['1stFlrSF'])


# 1stFlrSF resume:
#   *  The lineplot and the distplot show that there is no 1 to 1 correlation 
#   *  Numeric data
#   *  Normal distribution
#   
# Verdict:
#   *  Use this column in ML with "bins encoding"

# In[ ]:


plt1 = sns.lineplot(x="2ndFlrSF", y="SalePrice", data=df[['2ndFlrSF', 'SalePrice']])


# In[ ]:


plt2 = sns.distplot(df['2ndFlrSF'])


# 2ndFlrSF resume:
#   *  The lineplot and the distplot show that there is no 1 to 1 correlation 
#   *  Numeric data
#   
# Verdict:
#   *  Use this column in ML with "bins encoding"

# In[ ]:


plt1 = sns.lineplot(x="LowQualFinSF", y="SalePrice", data=df[['LowQualFinSF', 'SalePrice']])


# In[ ]:


plt2 = sns.distplot(df['LowQualFinSF'])


# LowQualFinSF resume:
#   *  The lineplot and the distplot show that there is no 1 to 1 correlation 
#   *  Numeric data
#   
# Verdict:
#   *  Use this column in ML with "bins encoding"

# In[ ]:


plt1 = sns.lineplot(x="GrLivArea", y="SalePrice", data=df[['GrLivArea', 'SalePrice']])


# In[ ]:


plt2 = sns.distplot(df['GrLivArea'])


# GrLivArea resume:
#   *  The lineplot and the distplot show that there is no 1 to 1 correlation 
#   *  Numeric data
#   *  Looks like a quite normal distribution
#   
# Verdict:
#   *  Use this column in ML with "bins encoding"

# In[ ]:


plt1 = sns.lineplot(x="BsmtFullBath", y="SalePrice", data=df[['BsmtFullBath', 'SalePrice']])


# In[ ]:


plt2 = sns.distplot(df['BsmtFullBath'])


# BsmtFullBath resume:
#   *  The lineplot and the distplot show that there is no 1 to 1 correlation 
#   *  Numeric data
#   
# Verdict:
#   *  Use this column in ML with "bins encoding"

# In[ ]:


plt1 = sns.lineplot(x="BsmtHalfBath", y="SalePrice", data=df[['BsmtHalfBath', 'SalePrice']])


# In[ ]:


plt2 = sns.distplot(df['BsmtHalfBath'])


# BsmtHalfBath resume:
#   *  The lineplot and the distplot show that there is no 1 to 1 correlation 
#   *  Numeric data
#   
# Verdict:
#   *  Use this column in ML with "bins encoding"

# In[ ]:


plt1 = sns.lineplot(x="FullBath", y="SalePrice", data=df[['FullBath', 'SalePrice']])


# In[ ]:


plt2 = sns.distplot(df['FullBath'])


# FullBath resume:
#   *  The lineplot and the distplot show that there is no 1 to 1 correlation 
#   *  Numeric data
#   
# Verdict:
#   *  Use this column in ML with "bins encoding"

# In[ ]:


plt1 = sns.lineplot(x="HalfBath", y="SalePrice", data=df[['HalfBath', 'SalePrice']])


# In[ ]:


plt2 = sns.distplot(df['HalfBath'])


# HalfBath resume:
#   *  The lineplot and the distplot show that there is no 1 to 1 correlation 
#   *  Numeric data
#   
# Verdict:
#   *  Use this column in ML with "bins encoding"

# In[ ]:


plt1 = sns.lineplot(x="BedroomAbvGr", y="SalePrice", data=df[['BedroomAbvGr', 'SalePrice']])


# In[ ]:


plt2 = sns.distplot(df['BedroomAbvGr'])


# BedroomAbvGr resume:
#   *  The lineplot and the distplot show that there is no 1 to 1 correlation 
#   *  Numeric data
#   
# Verdict:
#   *  Use this column in ML with "bins encoding"

# In[ ]:


plt1 = sns.lineplot(x="KitchenAbvGr", y="SalePrice", data=df[['KitchenAbvGr', 'SalePrice']])


# In[ ]:


plt2 = sns.distplot(df['KitchenAbvGr'])


# KitchenAbvGr resume:
#   *  Numeric data
#   
# Verdict:
#   *  Explore the column more precisely later (maybe we'll be able to drop this column)

# In[ ]:


plt1 = sns.barplot(x="KitchenQual", y="SalePrice", data=df[['KitchenQual', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['KitchenQual'])


# KitchenQual resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.lineplot(x="TotRmsAbvGrd", y="SalePrice", data=df[['TotRmsAbvGrd', 'SalePrice']])


# In[ ]:


plt2 = sns.distplot(df['TotRmsAbvGrd'])


# TotRmsAbvGrd resume:
#   *  The lineplot and the distplot show that there is no 1 to 1 correlation 
#   *  Numeric data
#   
# Verdict:
#   *  Use this column in ML with "bins encoding"

# In[ ]:


plt1 = sns.barplot(x="Functional", y="SalePrice", data=df[['Functional', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['Functional'])


# Functional resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.lineplot(x="Fireplaces", y="SalePrice", data=df[['Fireplaces', 'SalePrice']])


# In[ ]:


plt2 = sns.distplot(df['Fireplaces'])


# Fireplaces resume:
#   *  The lineplot and the distplot show that there is no 1 to 1 correlation 
#   *  Numeric data
#   
# Verdict:
#   *  Use this column in ML with "bins encoding"

# In[ ]:


plt1 = sns.barplot(x="FireplaceQu", y="SalePrice", data=df[['FireplaceQu', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['FireplaceQu'])


# FireplaceQu resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.lineplot(x="GarageYrBlt", y="SalePrice", data=df[['GarageYrBlt', 'SalePrice']])


# In[ ]:


plt2 = sns.distplot(df['GarageYrBlt'].dropna())


# GarageYrBlt resume:
#   *  The lineplot and the distplot show that there is no 1 to 1 correlation 
#   *  Numeric data
#   
# Verdict:
#   *  Use this column in ML with "bins encoding"

# In[ ]:


plt1 = sns.barplot(x="GarageFinish", y="SalePrice", data=df[['GarageFinish', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['GarageFinish'])


# GarageFinish resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.lineplot(x="GarageCars", y="SalePrice", data=df[['GarageCars', 'SalePrice']])


# In[ ]:


plt2 = sns.distplot(df['GarageCars'].dropna())


# GarageCars resume:
#   *  The lineplot and the distplot show that there is no 1 to 1 correlation 
#   *  Numeric data
#   
# Verdict:
#   *  Use this column in ML with "bins encoding"

# In[ ]:


plt1 = sns.lineplot(x="GarageArea", y="SalePrice", data=df[['GarageArea', 'SalePrice']])


# In[ ]:


plt2 = sns.distplot(df['GarageArea'].dropna())


# GarageArea resume:
#   *  The lineplot and the distplot show that there is no 1 to 1 correlation 
#   *  Numeric data
#   
# Verdict:
#   *  Use this column in ML with "bins encoding"

# In[ ]:


plt1 = sns.barplot(x="GarageQual", y="SalePrice", data=df[['GarageQual', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['GarageQual'])


# GarageQual resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.barplot(x="GarageCond", y="SalePrice", data=df[['GarageCond', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['GarageCond'])


# GarageCond resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.barplot(x="PavedDrive", y="SalePrice", data=df[['PavedDrive', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['PavedDrive'])


# PavedDrive resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.lineplot(x="WoodDeckSF", y="SalePrice", data=df[['WoodDeckSF', 'SalePrice']])


# In[ ]:


plt2 = sns.distplot(df['WoodDeckSF'].dropna())


# WoodDeckSF resume:
#   *  The lineplot and the distplot show that there is no 1 to 1 correlation 
#   *  Numeric data
#   
# Verdict:
#   *  Use this column in ML with "bins encoding"

# In[ ]:


plt1 = sns.lineplot(x="OpenPorchSF", y="SalePrice", data=df[['OpenPorchSF', 'SalePrice']])


# In[ ]:


plt2 = sns.distplot(df['OpenPorchSF'].dropna())


# OpenPorchSF resume:
#   *  The lineplot and the distplot show that there is no 1 to 1 correlation 
#   *  Numeric data
#   
# Verdict:
#   *  Use this column in ML with "bins encoding"

# In[ ]:


plt1 = sns.lineplot(x="EnclosedPorch", y="SalePrice", data=df[['EnclosedPorch', 'SalePrice']])


# In[ ]:


plt2 = sns.distplot(df['EnclosedPorch'].dropna())


# EnclosedPorch resume:
#   *  The lineplot and the distplot show that there is no 1 to 1 correlation 
#   *  Numeric data
#   
# Verdict:
#   *  Use this column in ML with "bins encoding"

# In[ ]:


plt1 = sns.lineplot(x="3SsnPorch", y="SalePrice", data=df[['3SsnPorch', 'SalePrice']])


# In[ ]:


plt2 = sns.distplot(df['3SsnPorch'].dropna())


# 3SsnPorch resume:
#   *  The lineplot and the distplot show that there is no 1 to 1 correlation 
#   *  Numeric data
#   
# Verdict:
#   *  Use this column in ML with "bins encoding"

# In[ ]:


plt1 = sns.lineplot(x="ScreenPorch", y="SalePrice", data=df[['ScreenPorch', 'SalePrice']])


# In[ ]:


plt2 = sns.distplot(df['ScreenPorch'].dropna())


# ScreenPorch resume:
#   *  The lineplot and the distplot show that there is no 1 to 1 correlation 
#   *  Numeric data
#   
# Verdict:
#   *  Use this column in ML with "bins encoding"

# In[ ]:


plt1 = sns.lineplot(x="PoolArea", y="SalePrice", data=df[['PoolArea', 'SalePrice']])


# In[ ]:


plt2 = sns.distplot(df['PoolArea'].dropna())


# PoolArea resume:
#   *  The lineplot and the distplot show that there is no 1 to 1 correlation 
#   *  Numeric data
#   
# Verdict:
#   *  Use this column in ML with "bins encoding"

# In[ ]:


plt1 = sns.barplot(x="PoolQC", y="SalePrice", data=df[['PoolQC', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['PoolQC'])


# PoolQC resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.barplot(x="Fence", y="SalePrice", data=df[['Fence', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['Fence'])


# Fence resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.barplot(x="MiscFeature", y="SalePrice", data=df[['MiscFeature', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['MiscFeature'])


# MiscFeature resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.lineplot(x="MiscVal", y="SalePrice", data=df[['MiscVal', 'SalePrice']])


# In[ ]:


plt2 = sns.distplot(df['MiscVal'].dropna())


# MiscVal resume:
#   *  The lineplot and the distplot show that there is no 1 to 1 correlation 
#   *  Numeric data
#   
# Verdict:
#   *  Use this column in ML with "bins encoding"

# In[ ]:


plt1 = sns.lineplot(x="MoSold", y="SalePrice", data=df[['MoSold', 'SalePrice']])


# In[ ]:


plt2 = sns.distplot(df['MoSold'].dropna())


# MoSold resume:
#   *  The lineplot and the distplot show that there is no 1 to 1 correlation 
#   *  Numeric data
#   
# Verdict:
#   *  Use this column in ML

# In[ ]:


plt1 = sns.lineplot(x="YrSold", y="SalePrice", data=df[['YrSold', 'SalePrice']])


# In[ ]:


plt2 = sns.distplot(df['YrSold'].dropna())


# YrSold resume:
#   *  The lineplot and the distplot show that there is no 1 to 1 correlation 
#   *  Numeric data
#   
# Verdict:
#   *  Use this column in ML

# In[ ]:


plt1 = sns.barplot(x="SaleType", y="SalePrice", data=df[['SaleType', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['SaleType'].dropna())


# SaleType resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# In[ ]:


plt1 = sns.barplot(x="SaleCondition", y="SalePrice", data=df[['SaleCondition', 'SalePrice']])


# In[ ]:


plt2 = sns.countplot(df['SaleCondition'].dropna())


# SaleCondition resume:
#   *  The barplot and the countplot show that there is no 1 to 1 correlation
#   *  Object data
#   
# Verdict:
#   *  Use this column in ML with one-hot encoding

# ## **Data quality plan:**
#   * MSSubClass - OK
#   * MSZoning - OK
#   * LotFrontage - MISSING 18% WITHOUT "NAN-MEANINGFUL"
#   *                     - 3rd percentile = 80 but max = 313
#   * LotArea - 3rd percentile = 11.601 but max = 215.245
#   * Street - OK
#   * Alley - MISSING 93% WITH! "NAN-MEANINGFUL"
#   * LotShape - OK
#   * LandContour - OK
#   * Utilities - OK
#   * LotConfig - OK
#   * LandSlope - OK
#   * Neighborhood - OK
#   * Condition1 - OK
#   * Condition2 - OK
#   * BldgType - OK
#   * HouseStyle - OK
#   * OverallQual- OK
#   * OverallCond - OK
#   * YearBuilt - 1rd percentile = 1954 but min = 1872
#   * YearRemodAdd - OK
#   * RoofStyle - OK
#   * RoofMatl - OK
#   * Exterior1st - OK
#   * Exterior2nd - OK
#   * MasVnrType - MISSING 0.5% WITH! "NAN-MEANINGFUL"
#   * MasVnrArea - MISSING 0.5% WITH! "NAN-MEANINGFUL"
#   *                     - 3rd percentile = 166 but max = 1600
#   * ExterQual - OK
#   * ExterCond- OK
#   * Foundation - OK
#   * BsmtQual - MISSING 0.5% WITH! "NAN-MEANINGFUL"
#   * BsmtCond - MISSING 0.5% WITH! "NAN-MEANINGFUL"
#   * BsmtExposure - MISSING 0.5% WITH! "NAN-MEANINGFUL"
#   * BsmtFinType1 - MISSING 0.5% WITH! "NAN-MEANINGFUL"
#   * BsmtFinSF1 - 3rd percentile = 712 but max = 1600
#   * BsmtFinType2 - MISSING 0.5% WITH! "NAN-MEANINGFUL
#   * BsmtFinSF2 - 3rd percentile = 0 but max = 1474
#   * BsmtUnfSF - 3rd percentile = 808 but max = 2336
#   * TotalBsmtSF - 3rd percentile = 1391 but max = 4692
#   * Heating - OK
#   * CentralAir - OK
#   * TotalBsmtSF - OK
#   * Electrical - missing only 1 object WITHOUT "NAN-MEANINGFUL"
#   * 1stFlrSF - 3rd percentile = 1391 but max = 4692
#   * 2ndFlrSF - 3rd percentile = 728 but max = 2065
#   * LowQualFinSF - 3rd percentile = 0 but max = 572
#   * GrLivArea - 3rd percentile = 1776 but max = 5642
#   * BsmtFullBath - OK
#   * BsmtHalfBath - OK
#   * FullBath - OK
#   * HalfBath - OK
#   * BedroomAbvGr - OK
#   * KitchenAbvGr - OK
#   * KitchenQual - OK
#   * TotRmsAbvGrd - OK
#   * Functional - OK
#   * Fireplaces- OK
#   * FireplaceQu - MISSING 47% WITH! "NAN-MEANINGFUL"
#   * GarageType - MISSING 0.05% WITH! "NAN-MEANINGFUL"
#   * GarageYrBlt - MISSING 0.05% WITH! "NAN-MEANINGFUL"
#   * GarageFinish - MISSING 0.05% WITH! "NAN-MEANINGFUL"
#   * GarageCars - OK
#   * GarageArea - 3rd percentile = 576 but max = 1418
#   * GarageQual - MISSING 0.05% WITH! "NAN-MEANINGFUL"
#   * GarageCond - MISSING 0.05% WITH! "NAN-MEANINGFUL"
#   * PavedDrive - OK
#   * WoodDeckSF - 3rd percentile = 168 but max = 857
#   * OpenPorchSF- 3rd percentile = 68 but max = 547
#   * EnclosedPorch- 3rd percentile = 0 but max = 552
#   * 3SsnPorch- 3rd percentile = 0 but max = 508
#   * ScreenPorch- 3rd percentile = 0 but max = 480
#   * PoolArea- 3rd percentile = 0 but max = 738
#   * PoolQC - MISSING 99% WITH! "NAN-MEANINGFUL"
#   * Fence - MISSING 99% WITH! "NAN-MEANINGFUL"
#   * MiscFeature - MISSING 99% WITH! "NAN-MEANINGFUL"
#   * MiscVal - 3rd percentile = 0 but max = 15500
#   * MoSold - OK
#   * YrSold - OK
#   * SaleType - OK
#   * SaleCondition - OK

# *We are going to label encode all object data and convert to data NAN / delete NAN to take Advance Data Excploration*

# In[ ]:


df["LotFrontage"] = df["LotFrontage"].dropna()
df["Electrical"] = df["Electrical"].dropna()
df = df.replace(np.nan, "EMPTY", regex=True)
df["MasVnrArea"] = df["MasVnrArea"].replace("EMPTY", -1)
df["GarageYrBlt"] = df["GarageYrBlt"].replace("EMPTY", -1)


# In[ ]:


df_test["LotFrontage"] = df_test["LotFrontage"].dropna()
df_test["Electrical"] = df_test["Electrical"].dropna()
df_test = df_test.replace(np.nan, "EMPTY", regex=True)
df_test["MasVnrArea"] = df_test["MasVnrArea"].replace("EMPTY", -1)
df_test["GarageYrBlt"] = df_test["GarageYrBlt"].replace("EMPTY", -1)


# In[ ]:


le = LabelEncoder()
categorical = list(df.select_dtypes(include=['object']).columns.values)
for cat in categorical:
    df[cat]=df[cat].astype('str')
    df[cat] = le.fit_transform(df[cat])


# In[ ]:


le = LabelEncoder()
categorical = list(df_test.select_dtypes(include=['object']).columns.values)
for cat in categorical:
    df_test[cat]=df_test[cat].astype('str')
    df_test[cat] = le.fit_transform(df_test[cat])


# In[ ]:


corr = df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# Now: equal-frequency binning (it will "repair" outliers a bit) and data normalization

# In[ ]:


for colName in df.columns:
    if(colName != "LotFrontage" and colName != "Electrical" and colName != "MasVnrArea" and colName != "GarageYrBlt" and colName != "Id" and colName != "SalePrice"):
        df[colName] = pd.qcut(df[colName].rank(method="first"), 5, labels=False)
        df_test[colName] = pd.qcut(df_test[colName].rank(method="first"), 5, labels=False)


# In[ ]:


df = df.drop(["Id"], 1)
maximum = df["SalePrice"].max()
minimum = df["SalePrice"].min()
    
scaler = MinMaxScaler(feature_range = (0,1))
scaled_df = scaler.fit_transform(df)
scaled_df_label = scaled_df[:, -1]
scaled_df = scaler.fit_transform(df.drop(["SalePrice"], 1))

df_id = df_test["Id"]
df_test = df_test.drop(["Id"], 1)

scaled_df_test = scaler.fit_transform(df_test)


# **END OF DATA EXPLORATION/DATA PREPORATION **
# 
# *It should be said that in place of one-hot encoding I used label because in my opinion one-hot encoding will cause dimension curse, but later we can try to use one-hot encoding.*

# # **MODELING STAGE**

# In[ ]:


# Regression Tree
#-----------------------------------------
from sklearn.tree import DecisionTreeRegressor
reg1 = DecisionTreeRegressor(criterion="mse", splitter="best", max_depth=6, min_samples_leaf=8, min_weight_fraction_leaf=0.1, max_features=21, max_leaf_nodes=19, min_impurity_decrease=2) 
#-----------------------------------------

# BaggingRegressor
#-----------------------------------------
from sklearn.ensemble import BaggingRegressor
reg2 = BaggingRegressor(n_estimators=500, max_samples=1000, max_features=35, bootstrap=False, bootstrap_features=False, warm_start=True)
#-----------------------------------------

# KNeighborsRegressor
#-----------------------------------------
from sklearn.neighbors import KNeighborsRegressor
reg3 =KNeighborsRegressor(n_neighbors=15, weights="distance", algorithm="auto", leaf_size=20, p=1)
#-----------------------------------------

# GaussianNB
#-----------------------------------------
from sklearn.naive_bayes import GaussianNB
reg4 = GaussianNB()
#-----------------------------------------

# linear_model: HuberRegressor
#-----------------------------------------
from sklearn.linear_model import HuberRegressor
reg5 = HuberRegressor(epsilon=1.35, max_iter=100000, alpha=0.0001, warm_start=False, fit_intercept=True, tol=1e-05)
#-----------------------------------------

# linear_model: PassiveAggressiveRegressor
#-----------------------------------------
from sklearn.linear_model import PassiveAggressiveRegressor
reg6 = PassiveAggressiveRegressor(C=1.0, average=False, early_stopping=False,
              epsilon=0.1, fit_intercept=True, loss="epsilon_insensitive",
              max_iter=100000, n_iter_no_change=5, random_state=0,
              shuffle=True, tol=0.001, validation_fraction=0.1,
              verbose=0, warm_start=False)
#-----------------------------------------

# linear_model: TheilSenRegressor
#-----------------------------------------
from sklearn.linear_model import TheilSenRegressor
reg7 = TheilSenRegressor()
#-----------------------------------------

# linear_model: BayesianRidge
#-----------------------------------------
from sklearn.linear_model import BayesianRidge
reg8 = BayesianRidge(n_iter=10000, tol=0.001, alpha_1=1e-11, alpha_2=0.1, lambda_1=0.1, lambda_2=1e-40, verbose=True)
#-----------------------------------------

# SVM
#-----------------------------------------
from sklearn.svm import SVR
reg9 = SVR(C=1, cache_size=200, coef0=0.0, degree=3, epsilon=0.01,
            gamma="scale", kernel="rbf", max_iter=-1, shrinking=True,
            tol=0.001, verbose=False)
#-----------------------------------------

# SGDRegressor
#-----------------------------------------
from sklearn.linear_model import SGDRegressor
reg10 = SGDRegressor(alpha=0.001, average=False, early_stopping=False, epsilon=0.001,
             eta0=0.01, fit_intercept=True, l1_ratio=0.15,
             learning_rate="adaptive", loss="squared_epsilon_insensitive",
             max_iter=5000, n_iter_no_change=5, penalty="elasticnet",
             power_t=0.25, random_state=None, shuffle=True, tol=0.001,
             validation_fraction=0.1, verbose=0, warm_start=False)
#-----------------------------------------

# MLPRegressor
#-----------------------------------------
from sklearn.neural_network import MLPRegressor
reg11 = reg = MLPRegressor(activation="tanh", alpha=0.001, batch_size=200, beta_1=0.9,
             beta_2=0.999, early_stopping=False, epsilon=1e-08,
             hidden_layer_sizes=(38, 15, 5, 1), learning_rate="constant",
             learning_rate_init=0.001, max_iter=200, momentum=0.9,
             n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
             random_state=None, shuffle=True, solver="lbfgs", tol=0.0001,
             validation_fraction=0.1, verbose=False, warm_start=False)
#-----------------------------------------

# GradientBoostingRegressor
#-----------------------------------------
from sklearn.ensemble import GradientBoostingRegressor
reg12 = GradientBoostingRegressor(learning_rate=0.15, n_estimators=500, max_depth = 3, min_samples_split = 2, min_samples_leaf = 3, max_features = 4, subsample = 0.95)
#-----------------------------------------


# Ensemble all models
#-----------------------------------------
from sklearn.ensemble import VotingRegressor

regs=[ #('1 Regression Tree', reg1),
     ('2 BaggingRegressor', reg2), 
     ('3 KNeighborsRegressor', reg3),
       #('4 GaussianNB', reg4),
     ('5 linear_model: HuberRegressor', reg5), 
     # ('6 linear_model: PassiveAggressiveRegressor', reg6),
     # ('7 linear_model: TheilSenRegressor', reg7), 
     ('8 linear_model: BayesianRidge', reg8), 
     ('9 svm', reg9),
     #('10 linear_model: SGDRegressor', reg10),
     #('11 MLPRegressor', reg11),
     #('12 GradientBoostingRegressor', reg12) 
     ] 

ensreg = VotingRegressor(estimators=regs)

Y = scaled_df_label
X = scaled_df

ensreg = ensreg.fit(X, Y)
prediction = ensreg.predict(scaled_df_test)

final_subm = []
for el in prediction:
    final_subm.append(el*(maximum - minimum) + minimum)
    
my_submission = pd.DataFrame({"Id": df_id, "SalePrice": final_subm})
my_submission.to_csv("prediction.csv", index=False)
#-----------------------------------------


# In[ ]:


# Cross-validation
#-----------------------------------------
from sklearn.model_selection import cross_val_score

scores = cross_val_score(ensreg, X, Y, cv=5)
print("Ensemble model")
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


for i in range(len(regs)):
    scores = cross_val_score(regs[i][1], X, Y, cv=5)
    print(regs[i][0])
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

