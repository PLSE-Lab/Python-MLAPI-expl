#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Required library imports
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


# Reading the training data
data = pd.read_csv('../input/train.csv')
data.head()


# > Let's see which columns have got null values

# In[ ]:


data.isnull().sum()


# > Can be seen from above that LotFrontage, Alley, FireplaceQu, PoolQC, Fence etc. are some variables having Null values.

# Before going further in analysing the data set, it is very much required to see how the target variable(i.e SalePrice) is behaving. With behaving we mean:
# * Distribution of SalePrice among various houses.
# * What's the max, min and average saleprice?
# * Visualise the SalePrice

# ### Target variable SalePrice -> Continuous feature

# In[ ]:


_min = data['SalePrice'].min()
print("Min Selling Price is :",_min)
_avg = data['SalePrice'].mean()
print("Average Selling Price is :",_avg)
_median = data['SalePrice'].median()
print("Median Selling Price is :",_median)
_max = data['SalePrice'].max()
print("Max Selling Price is :",_max)


# In[ ]:


data['SalePrice'].plot.hist(bins=30,edgecolor='black',color='green')
fig=plt.gcf()
fig.set_size_inches(25,15)
x_range = range(0,750000,25000)
plt.xticks(x_range)
plt.show()


# > SalePrice seems to follow a slightly skewed pattern from the standard Normal Distribution curve.

# > It becomes hard to comment on single values of a variable which is continuous in nature and can having too many values, that too analysing it in uni-variate fashion and commenting on a trend or general behaviour becomes tough.
# > So, here categorising the SalePrice(s) comes to rescue and will help us understand the behaviour of SalePrice in a much convinient way.

# In[ ]:


# Convert target variable to categorical like low,medium,high price and analyse
data['SalePrice_Cat'] = 0
data.loc[data.SalePrice <= 100000,'SalePrice_Cat'] = 0
data.loc[(data.SalePrice > 100000) & (data.SalePrice <=200000),'SalePrice_Cat'] = 1
data.loc[data.SalePrice > 300000,'SalePrice_Cat'] = 2
data.head(2)


# *Breakpoints of 100000, 200000 and 300000 is considered to be low, mid and high range, which is also clearly seen from the above histogram plot of SalePrice.*

# > Let's see how SalePrice behave in general.

# In[ ]:


# Binning based on domain
data['SalePrice_Cat'].value_counts()


# In[ ]:


sns.countplot('SalePrice_Cat',data=data,color='blue')
plt.show()


# *Looks like we have more houses falling in the mid range as compared to low and high*

# *Along with above categorisation as low, medium and high price, we can also look for quantile based segregation which buckets the data in buckets of equal number of houses.*

# It's very pretty amazing(:P) that we have that in pandas as qcut.

# In[ ]:


data['SalePrice_qcut_Cat'] = 0
data['SalePrice_qcut_Cat'] = pd.qcut(data['SalePrice'],20)
data.head(2)


# In[ ]:


# Binning based on quantiles
data['SalePrice_qcut_Cat'].value_counts()


# ##### MSSubClass: The building class -> Ordinal feature

# (20,30,40) - 1 story  
# (45,50) - 1 1/2 story  
# (60,70) - 2 story  

# In[ ]:


fig, axes = plt.subplots(1,2,figsize=(18,5))
sns.countplot('MSSubClass',data=data,ax=axes[0],color='blue')
sns.countplot('MSSubClass',hue='SalePrice_Cat',data=data,ax=axes[1])
plt.show()


# Also, can be clearly seen from the graphs as well that there is predominantly only one bar ie.  
# label 1 bar in MSSubClass >= 70.  

# In[ ]:


pd.crosstab([data.SalePrice_Cat],[data.MSSubClass],margins=True).style.background_gradient(cmap='summer_r')


# *It can be clearly seen from above that houses with MSSubClass >= 70 have higher  
# chances of having medium selling price within the range of 1l to 2l.  *

# In[ ]:


pd.crosstab([data.SalePrice_qcut_Cat],[data.MSSubClass],margins=True).style.background_gradient('summer_r')


# *Looks like nothing much is coming out from quantile based categorisation(or binning).*

# Let's also look on other variables like:
# * GrLivArea -> Continuous  
# * OverallQual -> Ordinal  
# * OverallCond -> Ordinal  
# * Neighborhood -> Categorical   
# *these look sensible. Let's see if they have got any information.*

# In[ ]:


# Checking for any null values
print(data['GrLivArea'].isnull().sum())
print(data['OverallQual'].isnull().sum())
print(data['OverallCond'].isnull().sum())
print(data['Neighborhood'].isnull().sum())
# There aren't any N.A values in above columns


# Let's also see their value counts individually.

# In[ ]:


# OverallQual
print("OverallQual")
print(data['OverallQual'].value_counts())


# In[ ]:


# OverallCond
print("OverallCond")
print(data['OverallCond'].value_counts())


# In[ ]:


# Neighborhood
print("Neighborhood")
print(data['Neighborhood'].value_counts())


# Let's take a look at above three features individually.

# #### OverallQual -> Ordinal Feature

# In[ ]:


pd.crosstab([data.SalePrice_Cat],[data.OverallQual],margins=True).style.background_gradient('summer_r')


# In[ ]:


fig, axes = plt.subplots(1,2,figsize=(18,5))
sns.countplot('OverallQual',data=data,ax=axes[0])
sns.countplot('OverallQual',hue='SalePrice_Cat',data=data,ax=axes[1])
plt.show()


# *It is clearly visible that overallQuality in range [1,3] and [9,10] has predominant towers of  
# lower and higher prices respectively.  
# i.e. All(predominantly) houses in overall Quality range [1,3] have lower prices while,  
# in the range [9,10] have higher prices.*

# #### OverallCond -> Ordinal feature

# In[ ]:


pd.crosstab([data.SalePrice_Cat],[data.OverallCond],margins=True).style.background_gradient('summer_r')


# In[ ]:


fig, axes = plt.subplots(1,2,figsize=(18,5))
sns.countplot('OverallCond',data=data,ax=axes[0])
sns.countplot('OverallCond',hue='SalePrice_Cat',data=data,ax=axes[1])
plt.show()


# #### Neighbourhood -> Categorical feature

# In[ ]:


pd.crosstab([data.SalePrice_Cat],[data.Neighborhood],margins=True).style.background_gradient('summer_r')


# In[ ]:


fig, axes = plt.subplots(1,2,figsize=(18,5))
sns.countplot('Neighborhood',data=data,ax=axes[0])
sns.countplot('Neighborhood',hue='SalePrice_Cat',data=data,ax=axes[1])
plt.show()


# *Looks some of neighborhoods have got only two bars and also, one or two single bars as well which can act as dominant but, even then the data density looks less there.  
# So, let's have a look at some other variables as well.*

# ##### GrLivArea -> Continuous Feature

# In[ ]:


_min_grliv_area = data['GrLivArea'].min()
print("Min GrLiveArea is :",_min_grliv_area)
_avg_grliv_area = data['GrLivArea'].mean()
print("Average GrLiveArea is :",_avg_grliv_area)
_median_grliv_area = data['GrLivArea'].median()
print("Median GrLiveArea is :",_median_grliv_area)
_max_grliv_area = data['GrLivArea'].max()
print("Max GrLiveArea is :",_max_grliv_area)


# In[ ]:


data['GrLivArea'].plot.hist(bins=20,edgecolor='black',color='yellow')
plt.xticks(range(0,5642,275))
plt.show()


# In[ ]:


data['GrLivArea_Cat'] = 0
data['GrLivArea_Cat'] = pd.qcut(data['GrLivArea'],10)
data.head()


# In[ ]:


data['GrLivArea_Cat'].value_counts()


# In[ ]:


pd.crosstab([data.GrLivArea_Cat],[data.SalePrice_Cat],margins=True).style.background_gradient('summer_r')


# It's clearly visible that as the living area increases, number of houses in   
# the higher pricing category increases and number of houses in the medium pricing category decreases.

# #### YearRemodAdd 

# *Let's see if there is any null value*

# In[ ]:


data['YearRemodAdd'].isnull().any()
# There isn't any N.A value in the column YearRemodAdd


# In[ ]:


print(data['YearRemodAdd'].min())
print(data['YearRemodAdd'].max())


# Let's again follow the easiest approach that we have been following and categorise *YearRemodAdd* to see how it behaves on aggregated level.

# In[ ]:


data['RemodelYear_Cat'] = 0
data.loc[data.YearRemodAdd <= 1950, 'RemodelYear_Cat'] = 0
data.loc[(data.YearRemodAdd > 1950) & (data.YearRemodAdd <= 1960), 'RemodelYear_Cat'] = 1
data.loc[(data.YearRemodAdd > 1960) & (data.YearRemodAdd <= 1970), 'RemodelYear_Cat'] = 2
data.loc[(data.YearRemodAdd > 1970) & (data.YearRemodAdd <= 1980), 'RemodelYear_Cat'] = 3
data.loc[(data.YearRemodAdd > 1980) & (data.YearRemodAdd <= 1990), 'RemodelYear_Cat'] = 4
data.loc[(data.YearRemodAdd > 1990) & (data.YearRemodAdd <= 2000), 'RemodelYear_Cat'] = 5
data.loc[(data.YearRemodAdd > 2000) & (data.YearRemodAdd <= 2010), 'RemodelYear_Cat'] = 6
data.head()


# In[ ]:


data['RemodelYear_Cat'].value_counts()


# In[ ]:


fig, axes = plt.subplots(1,2,figsize=(18,5))
sns.countplot('RemodelYear_Cat',data=data,ax=axes[0])
sns.countplot('RemodelYear_Cat',hue='SalePrice_Cat',data=data,ax=axes[1])
plt.show()


# *It can be seen from the graph that, Only Houses that have been re-modelled recently have got higher prices bar,  
# which is clearly shown in the right figure. *

# #### RoofStyle -> Categorical feature

# In[ ]:


data['RoofStyle'].isnull().any()
# No N.A value


# In[ ]:


data['RoofStyle'].value_counts()


# In[ ]:


data['RoofStyle'].value_counts().plot.pie()
plt.show()


# In[ ]:


fig, axes = plt.subplots(1,2,figsize=(18,5))
sns.countplot('RoofStyle',data=data,ax=axes[0])
sns.countplot('RoofStyle',hue='SalePrice_Cat',data=data,ax=axes[1])
plt.show()


# Nothing decisive is coming out of the variable RoofStyle.

# #### SaleCondition -> Categorical value

# In[ ]:


data['SaleCondition'].isnull().any()
# Again no N.A values


# In[ ]:


data['SaleCondition'].value_counts()


# In[ ]:


fig, axes = plt.subplots(1,2,figsize=(18,5))
sns.countplot('SaleCondition',data=data,ax=axes[0])
sns.countplot('SaleCondition',hue='SalePrice_Cat',data=data,ax=axes[1])
plt.show()


# #### YrSold

# In[ ]:


data['YrSold'].isnull().any()
# Again no N.A values


# In[ ]:


data['YrSold'].value_counts()


# In[ ]:


fig, axes = plt.subplots(1,2,figsize=(18,5))
sns.countplot('YrSold',data=data,ax=axes[0])
sns.countplot('YrSold',hue='SalePrice_Cat',data=data,ax=axes[1])
plt.show()


# #### LotShape -> Categorical Feature

# In[ ]:


data['LotShape'].value_counts()


# In[ ]:


fig, axes = plt.subplots(1,2,figsize=(18,5))
sns.countplot('LotShape',data=data,ax=axes[0])
sns.countplot('LotShape',hue='SalePrice_Cat',data=data,ax=axes[1])
plt.show()


# #### YearBuilt

# In[ ]:


data['YearBuilt'].isnull().any()
# No N.A. values, Great!


# In[ ]:


print(data['YearBuilt'].min())
print(data['YearBuilt'].max())
# data['YearBuilt'].value_counts()


# In[ ]:


data['YearBuilt_Cat'] = 0
data.loc[data.YearBuilt <= 1880,'YearBuilt_Cat'] = 0
data.loc[(data.YearBuilt > 1880) & (data.YearBuilt <= 1890),'YearBuilt_Cat'] = 1
data.loc[(data.YearBuilt > 1890) & (data.YearBuilt <= 2000),'YearBuilt_Cat'] = 2
data.loc[(data.YearBuilt > 2000) & (data.YearBuilt <= 2010),'YearBuilt_Cat'] = 3
data.loc[data.YearBuilt > 2010,'YearBuilt_Cat'] = 4
data.head(2)


# In[ ]:


fig, axes = plt.subplots(1,2,figsize=(18,5))
sns.countplot('YearBuilt_Cat',data=data,ax=axes[0],color='blue')
sns.countplot('YearBuilt_Cat',hue='SalePrice_Cat',data=data,ax=axes[1])
plt.show()


# Looks like nothing much informative is coming out of SaleCondition, YrSold, LotShape and YearBuilt too.

# *Some of the variables that has got some information/insight till now are:    
# **MSSubClass**, **OverallQuality**, **Living Area above ground**, **year of remodelling** and **neighborhood**(to some extent) 
# while, others havn't got much information to decide even the category of price the property falls into.*  
# **Note**:* Looking at some other notebooks it looks true too. (:P)*  
# ***Also***, regression approaches are coming soon.

# 
