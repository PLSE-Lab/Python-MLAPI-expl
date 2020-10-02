#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

iowa_file_path = '../input/train.csv'
iowa_data = pd.read_csv(iowa_file_path, index_col='Id')


# In[ ]:


#Dataset Fields 
print(iowa_data.columns)


# **Data fields Description**
# 
# *Here's a brief version of what you'll find in the data description file.*
# 
#     SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
#     
#     MSSubClass: The building class
#     MSZoning: The general zoning classification
#     
#     LotFrontage: Linear feet of street connected to property
#     LotArea: Lot size in square feet
#     Street: Type of road access
#     Alley: Type of alley access
#     LotShape: General shape of property
#     
#     LandContour: Flatness of the property
#     Utilities: Type of utilities available
#     LotConfig: Lot configuration
#     LandSlope: Slope of property
#     Neighborhood: Physical locations within Ames city limits
#     
#     Condition1: Proximity to main road or railroad
#     Condition2: Proximity to main road or railroad (if a second is present)
#     BldgType: Type of dwelling
#     HouseStyle: Style of 
#     
#     OverallQual: Overall material and finish quality
#     OverallCond: Overall condition rating
#     
#     YearBuilt: Original construction date
#     YearRemodAdd: Remodel date
#     
#     RoofStyle: Type of roof
#     RoofMatl: Roof material
#     
#     Exterior1st: Exterior covering on house
#     Exterior2nd: Exterior covering on house (if more than one material)
#     
#     MasVnrType: Masonry veneer type
#     MasVnrArea: Masonry veneer area in square feet
#     
#     ExterQual: Exterior material quality
#     ExterCond: Present condition of the material on the exterior
#     
#     Foundation: Type of foundation
#     
#     BsmtQual: Height of the basement
#     BsmtCond: General condition of the basement
#     BsmtExposure: Walkout or garden level basement walls
#     BsmtFinType1: Quality of basement finished area
#     BsmtFinSF1: Type 1 finished square feet
#     BsmtFinType2: Quality of second finished area (if present)
#     BsmtFinSF2: Type 2 finished square feet
#     BsmtUnfSF: Unfinished square feet of basement area
#     TotalBsmtSF: Total square feet of basement 
#     
#     Heating: Type of heating
#     HeatingQC: Heating quality and condition
#     CentralAir: Central air conditioning
#     Electrical: Electrical system
#     
#     1stFlrSF: First Floor square feet
#     2ndFlrSF: Second floor square feet
#     
#     LowQualFinSF: Low quality finished square feet (all floors)  
#     GrLivArea: Above grade (ground) living area square feet
#     BsmtFullBath: Basement full bathrooms
#     BsmtHalfBath: Basement half bathrooms
#     FullBath: Full bathrooms above grade
#     HalfBath: Half baths above grade
#     Bedroom: Number of bedrooms above basement level
#     
#     Kitchen: Number of kitchens
#     KitchenQual: Kitchen quality
#     
#     TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
#     Functional: Home functionality rating
#     
#     Fireplaces: Number of fireplaces
#     FireplaceQu: Fireplace quality
#     
#     GarageType: Garage location
#     GarageYrBlt: Year garage was built
#     GarageFinish: Interior finish of the garage
#     GarageCars: Size of garage in car capacity
#     GarageArea: Size of garage in square feet
#     GarageQual: Garage quality
#     GarageCond: Garage condition
#     
#     PavedDrive: Paved driveway
#     WoodDeckSF: Wood deck area in square feet
#     OpenPorchSF: Open porch area in square feet
#     EnclosedPorch: Enclosed porch area in square feet
#     3SsnPorch: Three season porch area in square feet
#     ScreenPorch: Screen porch area in square feet
#     
#     PoolArea: Pool area in square feet
#     PoolQC: Pool quality
#     
#     Fence: Fence quality
#     
#     MiscFeature: Miscellaneous feature not covered in other categories
#     MiscVal: Value of miscellaneous feature
#     
#     MoSold: Month Sold
#     YrSold: Year Sold
#     
#     SaleType: Type of sale
#     SaleCondition: Condition of sale

# In[ ]:


#Sample Values
iowa_data.describe()


# In[ ]:


#Data Fields Information
iowa_data.info()


# In[ ]:


for column in iowa_data:
    print(iowa_data[column].describe())


# In[ ]:


ax = iowa_data['SalePrice'].plot.hist(title ='Sale price distribution')
ax.set_xlabel("Property's Sale price ($)")
ax.set_ylabel("count")


# In[ ]:


#plt.hist(np.log(iowa_data.SalePrice), bins = 25)
ax = np.log(iowa_data.SalePrice).plot.hist(title ='Sale price distribution', fontsize=12)
ax.set_xlabel("log(SalePrice)")
ax.set_ylabel("count")


# In[ ]:


#Sale price and total living area comaprison 
fig = plt.figure()
plt.plot(iowa_data.GrLivArea, iowa_data.SalePrice,
         '.', alpha = 0.3)
fig.suptitle('SalePrice-GrLivArea', fontsize=18)
plt.xlabel('Above ground living area (sqft)', fontsize=14)
plt.ylabel("Property's Sale price ($)", fontsize=14)


# In[ ]:


#Sale price and Lot area in square foots comparison
fig = plt.figure()
plt.plot(iowa_data.LotArea, iowa_data.SalePrice,
         '.', alpha = 0.3)
fig.suptitle('SalePrice-LotArea', fontsize=18)
plt.xlabel('Lot size (sqft)', fontsize=14)
plt.ylabel("Property's Sale price ($)", fontsize=14)


# In[ ]:


#Sale price and Neighborhood in square foots comparison
fig = plt.figure()
plt.plot(iowa_data.Neighborhood, iowa_data.SalePrice,
         '.', alpha = 0.3)
fig.suptitle('SalePrice-Neighborhood', fontsize=18)
plt.xlabel('Neighborhood', fontsize=14)
plt.ylabel("Property's Sale price ($)", fontsize=14)
plt.xticks(rotation=90)


# In[ ]:


Neighborhood_meanSP =     iowa_data.groupby('Neighborhood')['SalePrice'].mean()
 
Neighborhood_meanSP = Neighborhood_meanSP.sort_values()

sns.pointplot(x = iowa_data.Neighborhood.values, y = iowa_data.SalePrice.values,
              order = Neighborhood_meanSP.index)
 
plt.xticks(rotation=70)


# In[ ]:


iowa_data['Neighborhood'].value_counts().plot.bar()


# In[ ]:


#Number of bedrooms distribution
sns.countplot(iowa_data.BedroomAbvGr)


# In[ ]:


#Total rooms distribution
sns.countplot(iowa_data.TotRmsAbvGrd)


# In[ ]:


#Type of foundation distribution
sns.countplot(iowa_data.Foundation)

