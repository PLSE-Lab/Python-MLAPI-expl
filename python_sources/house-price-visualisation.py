#!/usr/bin/env python
# coding: utf-8

# 

# # Introduction

# Here's a brief version of what you'll find in the data description file.
# 
# * SalePrice :the property's sale price in dollars. This is the target variable that you're trying to predict.
# * MSSubClass: The building class
# * MSZoning: The general zoning classification
# * LotFrontage: Linear feet of street connected to property
# * LotArea: Lot size in square feet
# * Street: Type of road access
# * Alley: Type of alley access
# * LotShape: General shape of property
# * LandContour: Flatness of the property
# * Utilities: Type of utilities available
# * LotConfig: Lot configuration
# * LandSlope: Slope of property
# * Neighborhood: Physical locations within Ames city limits
# * Condition1: Proximity to main road or railroad
# * Condition2: Proximity to main road or railroad (if a second is present)
# * BldgType: Type of dwelling
# * HouseStyle: Style of dwelling
# * OverallQual: Overall material and finish quality
# * OverallCond: Overall condition rating
# * YearBuilt: Original construction date
# * YearRemodAdd: Remodel date
# * RoofStyle: Type of roof
# * RoofMatl: Roof material
# * Exterior1st: Exterior covering on house
# * Exterior2nd: Exterior covering on house (if more than one material)
# * MasVnrType: Masonry veneer type
# * MasVnrArea: Masonry veneer area in square feet
# * ExterQual: Exterior material qualityFrom the above graph we can understand the curve have positive slope .ie, yearbuilt is positivly      varying on the SalesPrice.
# * ExterCond: Present condition of the material on the exterior
# * Foundation: Type of foundation
# * BsmtQual: Height of the basement
# * BsmtCond: General condition of the basement
# * BsmtExposure: Walkout or garden level basement walls
# * BsmtFinType1: Quality of basement finished area
# * BsmtFinSF1: Type 1 finished square feet
# * BsmtFinType2: Quality of second finished area (if present)
# * BsmtFinSF2: Type 2 finished square feet
# * BsmtUnfSF: Unfinished square feet of basement area 
# * TotalBsmtSF: Total square feet of basement area
# * Heating: Type of heating
# * HeatingQC: Heating quality and condition
# * CentralAir: Central air conditioningFrom the above graph we can understand the curve have positive slope .ie, yearbuilt is positivly varying on the SalesPrice.
# * Electrical: Electrical system
# * 1stFlrSF: First Floor square feet
# * 2ndFlrSF: Second floor square feet
# * LowQualFinSF: Low quality finished square feet (all floors)
# * GrLivArea: Above grade (ground) living area square feet
# * BsmtFullBath: Basement full bathrooms
# * BsmtHalfBath: Basement half bathrooms
# * FullBath: Full bathrooms above grade
# * HalfBath: Half baths above grade
# * Bedroom: Number of bedrooms above basement level
# * Kitchen: Number of kitchens
# * KitchenQual: Kitchen quality
# * TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
# * Functional: Home functionality rating
# * Fireplaces: Number of fireplaces
# * FireplaceQu: Fireplace quality
# * GarageType: Garage location
# * GarageYrBlt: Year garage was built
# * GarageFinish: Interior finish of the garage
# * GarageCars: Size of garage in car capacity
# * GarageArea: Size of garage in square feet
# * GarageQual: Garage quality
# * GarageCond: Garage condition
# * PavedDrive: Paved driveway
# * WoodDeckSF: Wood deck area in square feet
# * OpenPorchSF: Open porch area in square feet
# * EnclosedPorch: Enclosed porch area in square feet
# * 3SsnPorch: Three season porch area in square feet
# * ScreenPorch: Screen porch area in square feet
# * PoolArea: Pool area in square feet
# * PoolQC: Pool quality
# * Fence: Fence quality
# * MiscFeature: Miscellaneous feature not covered in other categories
# * MiscVal: $Value of miscellaneous feature
# * MoSold: Month Sold
# * YrSold: Year Sold
# * SaleType: Type of sale
# * SaleCondition: Condition of sale
# 
# 
# 
# 
# 
# 
# 

# #  EDA and data analysis ,making plots and trying to come up with data analysis outcomes.

# Setting the working directory.Other wise we can't read the files.It may show the error

# In[ ]:


import os
os.getcwd()
os.chdir('/kaggle/input/houseprice')


# In[ ]:


get_ipython().system('pip install missingno #INSTALLING MISSINGNO LIBRARY')


# # **Import Libraries**

# In[ ]:


import numpy as np  # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt #for plotting
import seaborn as sns  #Data visualisation
import missingno as msno 


# In[ ]:


#reading the dataframe
df=pd.read_csv("train.csv",index_col='Id')


# To take a closer look at the data itself. With the help of the head() and tail() functions of the Pandas library, you can easily check out the first and last lines of your DataFrame

# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.info()
#it will give the overall idea about the dataframe 


# In[ ]:


df.describe()
# getting a basic description of your data


# In[ ]:


df.shape #it will give the count of number of rows and coloumns in the dataframe


# In[ ]:


#Let's look at the skewness of our dataset

df.skew()


# In[ ]:


df.dtypes.value_counts() #it will the data types and its count


# In[ ]:


#Let us examine numerical features in the train dataset
#numeric_features = df.select_dtypes(include=[np.number])
#numeric_features.columns

num_col=df.select_dtypes(exclude='object')
cat_col=df.select_dtypes(exclude=['int64','float64'])


# In[ ]:


#Let us examine categorical features in the train dataset


categorical_features = df.select_dtypes(include=[np.object])
categorical_features.columns
#categorical_features.shape


# # Misssing Value

# In[ ]:


msno.heatmap(df)


# Abovefrom the heat map we can find that MasVnrType is not highly correlated with another feature 

# In[ ]:


# HEATMAP TO SEE MISSING VALUES
plt.figure(figsize=(15,5))
sns.heatmap(num_col.isnull(),yticklabels=0,cbar=False,cmap='viridis')


# In[ ]:


#heatmap shows LotFrontage,MasVnrArea,GarageYrBlt have the missing values


# In[ ]:



#missing value
num_col.isnull().sum()


# In[ ]:


#Datacleaning
#so here we want deal with missing values and replace all nun values of column
#mainly LotFrontage,MasVnrArea,GarageYrBlt


# In[ ]:


#correlation map
f,ax = plt.subplots(figsize=(20,2))
sns.heatmap(num_col.corr().iloc[7:8,:], annot=True, linewidths=.8, fmt= '.1f',ax=ax)
# this shows that MasVnrArea is not highly corelated to any other feature


# In[ ]:




sns.kdeplot(num_col.MasVnrArea,Label='MasVnrArea',color='b');


# In[ ]:


#so most of the values is near by 0 so we can replace the nan value with 0Analysis of features against sale price


# In[ ]:


num_col.MasVnrArea.replace({np.nan:0},inplace=True)
num_col


# In[ ]:


f,ax = plt.subplots(figsize=(20,2))
sns.heatmap(num_col.corr().iloc[1:2,:], annot=True, linewidths=.8, fmt= '.1f',ax=ax)


# In[ ]:


#lotFrontage has enough corelation with sales price so we cant drop it


# In[ ]:


num_col.LotFrontage.describe()


# In[ ]:


num_col['LotFrontage'].replace({np.nan:num_col.LotFrontage.mean()},inplace=True)


# In[ ]:


sns.scatterplot(x=num_col['LotFrontage'],y=num_col['LotArea'],hue=num_col['SalePrice']);


# From the above graph we can understand that  sales price is increasing more when the Lot frontage is increasing.the lot area is not making that  much variation.The houses having LotArea in between 0-50000 is some part depending the Lot Frontage   

# In[ ]:


sns.scatterplot(x=num_col['LotFrontage'],y=num_col['1stFlrSF'],hue=num_col['SalePrice']);


# From the above scatter plot it is clear that if the house have both more 1st floor squre feet and LotFrontage then they have High sale price

# In[ ]:


sns.scatterplot(x=num_col['LotFrontage'],y=num_col['TotalBsmtSF'],hue=num_col['SalePrice']);


# From the above scatter plot it is clear that if the house have both more  and LotFrontage then they have High sale price

# In[ ]:


sns.scatterplot(x=num_col['LotFrontage'],y=num_col['GarageArea'],hue=num_col['SalePrice']);


# If the garage area and Lotfrontage [50-150] we can find that price is high while the garage area is high but there are less percent exceptional cases also

# In[ ]:


sns.scatterplot(x=num_col['LotFrontage'],y=num_col['BedroomAbvGr'],hue=num_col['SalePrice']);


# # Analysis of features against sale price

# In[ ]:


#Creating a heat map of all the numerical features.
plt.figure(figsize=(20,20))
mat = np.round(num_col.corr(), decimals=2)
sns.heatmap(data=mat, linewidths=1, linecolor='black');


# 

# In[ ]:


#Getting features that have a correlation value greater than 0.5 against sale price.
for val in range(len(mat['SalePrice'])):
    if abs(mat['SalePrice'].iloc[val]) > 0.5:
        print(mat['SalePrice'].iloc[val:val+1]) 


# OverallQual is highly correlated with target feature SalePrice 0.79 can you see. we'll see how it effected the saleprice in below graph.

# In[ ]:


sns.barplot(df.OverallQual,df.SalePrice);


# OverallQual is directly propostional to the sales price.The meterials used and the finishing quality is a major factor depend on the sales price. from the above graph we can easily understand that.

# 

# # BOX PLOT

# when we look for the house,we surely look for the street.it is importent criteria to validate the price
# 

# In[ ]:


sns.boxplot(y=df['SalePrice'],x=df['Street'],palette='rainbow');


# From the above we can understand the street 'pave' have more sales price comparing to Grvl.may be the factors of that street will be good on another

# In[ ]:


sns.boxplot(y=df['SalePrice'],x=df['LotShape'],palette='rainbow');


# SalesPrice  is depending the Genaral shape of property  IR2 has   high median and Reg has lowest MEDIAN

# In[ ]:


df.Electrical.replace({np.nan:df['Electrical'].mode()},inplace=True)


# In[ ]:


df.Electrical.unique()


# In[ ]:


sns.countplot(df['Electrical']);


# Since there is only one missing value here, we replace it with SBrkr

# # lmplot

# In[ ]:


sns.lmplot(data= df,x='SalePrice',y='YearBuilt',scatter=False);


# From the above graph we can understand the curve have positive slope .ie, yearbuilt is positivly varying on the SalesPrice.

# In[ ]:


sns.lmplot(data= df,x='SalePrice',y='YearRemodAdd',scatter=False);


# From the above graph we can understand the curve have positive slope .ie, yearbuilt is positivly varying on the SalesPrice.

# In[ ]:


sns.lmplot(data= df,x='SalePrice',y='FullBath',scatter=False);


# From the above graph we can understand the curve have positive slope .ie, FullBath is positivly varying on the SalesPrice.

# In[ ]:




