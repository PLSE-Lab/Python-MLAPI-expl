#!/usr/bin/env python
# coding: utf-8

# 
# **Description**
# 
# Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.
# 
# With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges  to predict the final price of each home.
# 
# **Goal**
# 
# To predict the sales price for each house. For each Id in the test set,  predict the value of the SalePrice variable. 
# 
# **Steps**
# 
# * Load Libraries and view Data
# * Data Visualization
# * Feature Engineering
# * Create a model
# * Submit
# 
# 
# **Load Libraries and view Data**

# In[ ]:


#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
from scipy.stats import norm, skew #for some statistics
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import plotly.tools as tls

#Load datasets

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head(2) #Check the dataset


# In[ ]:


train.shape 


# As we see in the dataset, we have about 81 different columns. We will try to predict the targer variable - "SalePrice"  which is the property's sale price in dollars.
# 

# In[ ]:


train.describe(include="all")


# In[ ]:


#Correlation

train.corr()["SalePrice"]


# We observe that below features have significant correlation to "SalePrice"
# 
# *  **OverallQual**:  Overall material and finish quality
# * **GrLivArea**: Above grade (ground) living area square feet
# * **GarageCars**: Size of garage in car capacity
# * **GarageArea**: Size of garage in square feet
# * **GarageYrBlt**: Year garage was built
# * **TotalBsmtSF**:  Total square feet of basement area
# * **1stFlrSF**: First Floor square feet
# * **FullBath**: Full bathrooms above grade
# * **TotRmsAbvGrd**:  Total rooms above grade (does not include bathrooms)
# * **YearBuilt**:  Original construction date
# * **YearRemodAdd**:  Remodel date
# 
# Heatmaps are great to visualize these kind of correlations. Let's see one for these variables

# In[ ]:


corr=train[["SalePrice","OverallQual","GrLivArea","GarageCars",
                  "GarageArea","GarageYrBlt","TotalBsmtSF","1stFlrSF","FullBath",
                  "TotRmsAbvGrd","YearBuilt","YearRemodAdd"]].corr()

sns.set(font_scale=1.15)
plt.figure(figsize=(14, 10))

sns.heatmap(corr, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='YlGnBu',linecolor="black")
plt.title('Correlation between features');


# 
# 
# Some thoughts: 
# 
# * 'GarageCars' and 'GarageArea' are similar. So we can just keep one. We will keep 'GarageCars' as the correlation is higher with SalePrice
# *  'TotalBsmtSF' and '1stFloor'  are similar.  We will keep  'TotalBsmtSF' as the correlation is *slightly* higher with SalePrice
# * TotRmsAbvGrd' and 'GrLivArea' are similar.  We will keep 'GrLivArea' as the correlation is higher with SalePrice
# 
# Let's proceed and visualize some pairplots for these variables

# In[ ]:


sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size = 2.5)
plt.show();


# In[ ]:


#More Data visualizations
plt.figure(figsize=(12,6))
plt.scatter(x='GrLivArea', y='SalePrice', data=train)
plt.xlabel("GrLivArea", fontsize=13)
plt.ylabel("SalePrice", fontsize=13)
plt.ylim(0,800000)


# In[ ]:


agg = train['MSZoning'].value_counts()[:10]
labels = list(reversed(list(agg.index )))
values = list(reversed(list(agg.values)))

trace1 = go.Pie(labels=labels, values=values, marker=dict(colors=['red']))
layout = dict(title='The general zoning classification', legend=dict(orientation="h"));


fig = go.Figure(data=[trace1], layout=layout)
iplot(fig, filename='stacked-bar')



# In[ ]:


sns.violinplot(x='FullBath', y='SalePrice', data=train)
plt.title("Sale Price vs Full Bathrooms")


# In[ ]:


sns.violinplot( x="HalfBath",y="SalePrice", data=train)
plt.title("Sale Price vs Half Bathrooms");


# In[ ]:



#1st Floor in sq.feet
plt.scatter(train["1stFlrSF"],train.SalePrice, color='red')
plt.title("Sale Price vs. First Floor square feet")
plt.ylabel('Sale Price (in dollars)')
plt.xlabel("First Floor square feet");


# In[ ]:


plt.figure(figsize=(14,6))
plt.xticks(rotation=60) 
sns.barplot(x="Neighborhood", y = "SalePrice", data=train)
plt.title("Sale Price vs Neighborhood",fontsize=15 )


# In[ ]:


plt.figure(figsize=(14,6))
sns.barplot(x="TotRmsAbvGrd",y="SalePrice",data=train)
plt.title("Sale Price vs Number of rooms", fontsize=15);


# In[ ]:


plt.figure(figsize=(14,6))
sns.barplot(x="OverallQual",y="SalePrice",data=train)
plt.title("Sale Price vs 'Overall material and finish quality'", fontsize=15);


# **Missing Data**
# 
# Data can have missing values for a number of reasons such as observations that were not recorded and data corruption.
# 
# Handling missing data is important as many machine learning algorithms do not support data with missing values.

# In[ ]:


#missing data
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# 
# Anything above 25% we will drop. So we will drop 'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'. 
# 
# * PoolQC : Pool quality (That make sense, given the huge ratio of missing value (+99%) and majority of houses have no Pool at all in general)
# * MiscFeature : Miscellaneous feature not covered in other categories
# * Alley : Type of alley access
# * Fence : Fence quality
# * FireplaceQu : Fireplace quality

# In[ ]:


train = train.drop(columns = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis = 1)


# In[ ]:


train.shape


# In[ ]:


#LotFrontage : Since the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood , we can fill in missing values by the median LotFrontage of the neighborhood

train["LotFrontage"] = train.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))


# In[ ]:


#GarageType, GarageFinish, GarageQual and GarageCond : Replacing missing data with None

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    train[col] = train[col].fillna('None')


# In[ ]:


#GarageYrBlt, GarageArea and GarageCars : Replacing missing data with 0 (Since No garage = no cars in such garage.)

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    train[col] = train[col].fillna(0)
    
#BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath : missing values are likely zero for having no basement

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    train[col] = train[col].fillna(0)
    
#BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2 : for all these categorical basement-related features, NaN means that there is no basement.

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    train[col] = train[col].fillna('None')

#MasVnrArea and MasVnrType : NA most likely means no masonry veneer for these houses. We can fill 0 for the area and None for the type.

train["MasVnrType"] = train["MasVnrType"].fillna("None")
train["MasVnrArea"] = train["MasVnrArea"].fillna(0)


#MSZoning (The general zoning classification) : 'RL' is by far the most common value. So we can fill in missing values with 'RL'

train['MSZoning'] = train['MSZoning'].fillna(train['MSZoning'].mode()[0])

#Utilities : For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . 
#Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling. We can remove it

train = train.drop(['Utilities'], axis=1)

#Electrical : It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.

train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])


# In[ ]:


#Check remaining missing values if any 
train_na = (train.isnull().sum() / len(train)) * 100
train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :train_na})
missing_data.head()


# Next Steps:
# 
# * Feature Engineering
# * Creating and submitting model
# 
# *Please upvote or comment if you liked the kernel :) *
