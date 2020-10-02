#!/usr/bin/env python
# coding: utf-8

# # Data Cleaning

# <pre>here i am going to clean data or fill NaN values.
# understanding all coulumns and then apply best strategy for filling null values.
# so lets start<pre/>
# *if you find anything helpful please upvote*

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# reading data
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")


# In[ ]:


train.shape, test.shape


# there are 18 features has null rows 

# In[ ]:


train.isna().sum().sort_values()[-19:-1]


# in the test set 32 columns has null values

# In[ ]:


test.isna().sum().sort_values()[-33:-1]


# ### 1)MiscFeature: Miscellaneous feature not covered in other categories
# here null values means,NA means the house does not have any features so,simply NA is a one type of category so,filled as 'NA'.belows categories of MiscFeature Columns
# 
#            Elev	Elevator
#            Gar2	2nd Garage (if not described in garage section)
#            Othr	Other
#            Shed	Shed (over 100 SF)
#            TenC	Tennis Court
#            NA	None 
# ### 2)Alley: Type of alley access to property
# here null values means,NA means the house does not have alley access so,simply NA is a one type of category so,filled as 'NA'.belows categories of Alley Columns
# 
#            Grvl	Gravel
#            Pave	Paved
#            NA 	No alley access
# ### 3)FireplaceQu: Fireplace quality
# here null values means,NA means the house does not have Fireplace so,simply NA is a one type of category so,filled as 'NA'. belows categories of FireplaceQu Columns
# 
#            Ex	Excellent - Exceptional Masonry Fireplace
#            Gd	Good - Masonry Fireplace in main level
#            TA	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
#            Fa	Fair - Prefabricated Fireplace in basement
#            Po	Poor - Ben Franklin Stove
#            NA	No Fireplace
# ### 4)Fence: Fence quality
# here null values means,NA means the house does not have Fence so,simply NA is a one type of category so,filled as 'NA'.belows categories of Fence Columns
# 
#            GdPrv	Good Privacy
#            MnPrv	Minimum Privacy
#            GdWo	Good Wood
#            MnWw	Minimum Wood/Wire
#            NA	No Fence
# ### 4)PoolQC: Pool quality <- same as above, NaN = No pool
# 		
#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        NA	No Pool	      

# In[ ]:


#Filling with "NA" string
for col in ['Alley','FireplaceQu','Fence','MiscFeature','PoolQC']:
    train[col].fillna('NA', inplace=True)
    test[col].fillna('NA', inplace=True)


#  ### 5) LotFrontage: Linear feet of street connected to property

# <pre>This column is categorical columns but it has numerical values.so i'm gonna fill this values with most frequent value of the column, below code return most frequent value.<pre/> 
# train["LotFrontage"].value_counts().to_frame().index[0]

# In[ ]:


train['LotFrontage'].value_counts()


# In[ ]:


train['LotFrontage'].fillna(train["LotFrontage"].value_counts().to_frame().index[0], inplace=True)
test['LotFrontage'].fillna(test["LotFrontage"].value_counts().to_frame().index[0], inplace=True)


# ### In below 6-9 columns,null value means there is no Garage, because these columns has same no of missing values and with same rows.so ican say that null values means No Garage. 

# 		
# ### 6)GarageType: Garage location
# 		
#        2Types	More than one type of garage
#        Attchd	Attached to home
#        Basment	Basement Garage
#        BuiltIn	Built-In (Garage part of house - typically has room above garage)
#        CarPort	Car Port
#        Detchd	Detached from home
#        NA	No Garage
# ### 7)GarageFinish: Interior finish of the garage
# 
#        Fin	Finished
#        RFn	Rough Finished	
#        Unf	Unfinished
#        NA	No Garage
# ### 8)GarageQual: Garage quality
# 
#        Ex	Excellent
#        Gd	Good
#        TA	Typical/Average
#        Fa	Fair
#        Po	Poor
#        NA	No Garage
# ### 9)GarageCond: Garage condition
# 
#        Ex	Excellent
#        Gd	Good
#        TA	Typical/Average
#        Fa	Fair
#        Po	Poor
#        NA	No Garage
#        
#  ### 10)GarageYrBlt: Year garage was built <- if there is no garage then not possible build year so , filling with NA

# In[ ]:


train[['GarageQual','GarageFinish','GarageYrBlt','GarageType','GarageCond']].isna().head(7)


# In[ ]:


for col in ['GarageQual','GarageFinish','GarageYrBlt','GarageType','GarageCond']:
    train[col].fillna('NA',inplace=True)
    test[col].fillna('NA',inplace=True)


# ### for belows columns applying same strategy of Garage,null values means No Basement.

# ### 11)BsmtQual: Evaluates the height of the basement
# 
#        Ex	Excellent (100+ inches)	
#        Gd	Good (90-99 inches)
#        TA	Typical (80-89 inches)
#        Fa	Fair (70-79 inches)
#        Po	Poor (<70 inches
#        NA	No Basement		
# ### 12)BsmtCond: Evaluates the general condition of the basement
# 
#        Ex	Excellent
#        Gd	Good
#        TA	Typical - slight dampness allowed
#        Fa	Fair - dampness or some cracking or settling
#        Po	Poor - Severe cracking, settling, or wetness
#        NA	No Basement	
# ### 13)BsmtExposure: Refers to walkout or garden level walls
# 
#        Gd	Good Exposure
#        Av	Average Exposure (split levels or foyers typically score average or above)	
#        Mn	Mimimum Exposure
#        No	No Exposure
#        NA	No Basement	
# ### 14)BsmtFinType1: Rating of basement finished area
# 
#        GLQ	Good Living Quarters
#        ALQ	Average Living Quarters
#        BLQ	Below Average Living Quarters	
#        Rec	Average Rec Room
#        LwQ	Low Quality
#        Unf	Unfinshed
#        NA	No Basement
# ### 15)BsmtFinType2: Rating of basement finished area (if multiple types)
# 
#        GLQ	Good Living Quarters
#        ALQ	Average Living Quarters
#        BLQ	Below Average Living Quarters	
#        Rec	Average Rec Room
#        LwQ	Low Quality
#        Unf	Unfinshed
#        NA	No Basement

# In[ ]:


for col in ['BsmtQual','BsmtCond','BsmtFinType1','BsmtFinType2','BsmtExposure']:
    train[col].fillna('NA',inplace=True)
    test[col].fillna('NA',inplace=True)


# ### 16)Electrical: Electrical system
# 
#        SBrkr	Standard Circuit Breakers & Romex
#        FuseA	Fuse Box over 60 AMP and all Romex wiring (Average)	
#        FuseF	60 AMP Fuse Box and mostly Romex wiring (Fair)
#        FuseP	60 AMP Fuse Box and mostly knob & tube wiring (poor)
#        Mix	Mixed

# ### in this column, maximum value has 'SBrkr' category.so simply fillna with 'SBrkr'.

# In[ ]:


train['Electrical'].value_counts()


# In[ ]:


train['Electrical'].fillna('SBrkr',inplace=True)


# ### Belows bunch of columns has 1,2,4 missing values.so i'm gonna use same strategy for all columns.

# In[ ]:


missings = ['GarageCars','GarageArea','KitchenQual','Exterior1st','SaleType','TotalBsmtSF','BsmtUnfSF','Exterior2nd',
            'BsmtFinSF1','BsmtFinSF2','BsmtFullBath','Functional','Utilities','BsmtHalfBath','MSZoning']
train[missings].head()


# dereferencing numerical and categorical columns.

# In[ ]:


numerical=['GarageCars','GarageArea','TotalBsmtSF','BsmtUnfSF','BsmtFinSF1','BsmtFinSF2','BsmtFullBath','BsmtHalfBath']
categorical = ['KitchenQual','Exterior1st','SaleType','Exterior2nd','Functional','Utilities','MSZoning']


# for numerical columns filling NaN as median value.

# In[ ]:


# using Imputer class of sklearn libs.
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy='median',axis=0)
imputer.fit(test[numerical] + train[numerical])
test[numerical] = imputer.transform(test[numerical])
train[numerical] = imputer.transform(train[numerical])


# for categorical columns filling NaN with most frequent value of the column.

# In[ ]:


for i in categorical:
    train[i].fillna(train[i].value_counts().to_frame().index[0], inplace=True)
    test[i].fillna(test[i].value_counts().to_frame().index[0], inplace=True)    


# ### 17)MasVnrType: Masonry veneer type
# 
#        BrkCmn	Brick Common
#        BrkFace	Brick Face
#        CBlock	Cinder Block
#        None	None
#        Stone	Stone
#  carefully here Null value means not "None", because the column(17) has already 'None's. value	
# ### 18)MasVnrArea: Masonry veneer area in square feet <- for this column filling NaN with median value if 'saleprice' which 'masVnrType' in 'BrkFace' category. look below code you will understand.

# In[ ]:


train[train['MasVnrType'].isna()][['SalePrice','MasVnrType','MasVnrArea']]


# In[ ]:


print(train[train['MasVnrType']=='None']['SalePrice'].median())
print(train[train['MasVnrType']=='BrkFace']['SalePrice'].median())
print(train[train['MasVnrType']=='Stone']['SalePrice'].median())
print(train[train['MasVnrType']=='BrkCmn']['SalePrice'].median())


# In[ ]:


train['MasVnrArea'].fillna(181000,inplace=True)
test['MasVnrArea'].fillna(181000,inplace=True)

train['MasVnrType'].fillna('NA',inplace=True)
test['MasVnrType'].fillna('NA',inplace=True)


# > that's it, there no missing values

# In[ ]:


print(train.isna().sum().sort_values()[-5:-1])
print(test.isna().sum().sort_values()[-5:-1])


# saving files (optional)

# In[ ]:


# train.to_csv('new_train',index=False)
# test.to_csv('new_test',index=False)


# ## Thank you, upvote if you like
