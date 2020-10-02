#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from scipy.stats import pearsonr,norm, probplot
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')


# In[ ]:


train_df.columns


# In[ ]:


train_df['SalePrice'].describe()


# In[ ]:


#histogram
sns.distplot(train_df['SalePrice'])


# - Deviate from normal distribution
# - Skewness
# - Peakedness

# In[ ]:


print("Skewness: %f" % train_df['SalePrice'].skew())
print("Kurtosis: %f" % train_df['SalePrice'].kurt())


# In[ ]:


#Scatter plot grlivarea/Saleprice
var = 'GrLivArea'
data = pd.concat([train_df['SalePrice'],train_df[var]],axis=1)
data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000))


# In[ ]:


gLivArea_SalePrice_corr = pearsonr(train_df['SalePrice'],
                                  train_df[var])
gLivArea_SalePrice_corr


# The correlation seems to be very strong

# In[ ]:


# scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([train_df['SalePrice'],train_df[var]],axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))


# In[ ]:


totalBsmSF_SalePrice_corr = pearsonr(train_df['SalePrice'],
                                    train_df[var])
totalBsmSF_SalePrice_corr


# Positive correlation but some times it is 0 and the price is non-zero. Could this a data issue ?

# Now let us have a look at some categorical variables

# In[ ]:


#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([train_df['SalePrice'], train_df[var]],axis=1)
f, ax = plt.subplots(figsize=(8,6))
fig = sns.boxplot(x=var, y='SalePrice', data = data)
fig.axis(ymin=0, ymax=800000)


# Sales Price shows a good response to the overall  quality variable.

# In[ ]:


var = 'YearBuilt'
data = pd.concat([train_df['SalePrice'],train_df[var]], axis=1)
f, ax = plt.subplots(figsize=(16,8))
fig = sns.boxplot(x=var, y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)


# Not very heavily implied but newly built houses seem to have a higher price. 

# **Variable Summary**
# * We have just anlyzed four variables and see some string linear relation with continuous variables
# * For categorical variables the overall quality seems to have significant impact on the house price

# **Now let's look at the data in a different way**

# In[ ]:


#Correlation matrix heat map style
corrmat = train_df.corr()
f,ax = plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=.8,square=True)


# * We see a strong correlation between Garage variables signaling a situation of multicollinearity. We can can conclude that they give almost same information
# * Same holds true for TotalBsmtSF and 1srFlrSF
# * We also see that the variables we analyzed are holding on linearity in the heatmap

# In[ ]:


# Correlation matrix  zoomed heat map style
k = 10 # No of variables for heat map
cols = corrmat.nlargest(k,'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':10},
                 yticklabels=cols.values,xticklabels=cols.values)
plt.show()


# * OverallQual', 'GrLivArea' and 'TotalBsmtSF' are strongly correlated with 'SalePrice'
# * 'GarageCars' and 'GarageArea' are strongly related and are like twin sisters. We only need of the variables. We will use GarageCars as it is higher
# * As TotalBsmtSF and 1stFloor are also stroing correlated we will keep one
# * FullBath seems to show should good relation

# Let's plot scatter plots between sales price and the chosen variables

# In[ ]:


sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train_df[cols],size=2.5)
plt.show()


# The plots confirm our findings. One interesting relation that jumps out is between GrLivArea and TotalBmstSF. It makes sense that the living area will be more that the basement area other than you are trying to build a nuclear bunker 

# **Now we come at the major section of missing data**
# * How bad is the missing data situation ?
# * Does the missing data have a pattern or is random ?

# In[ ]:


total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis=1,keys=['Total','Percent'])
missing_data.head(20)


# **Let's handle missing data now**
# * Any variable that has more than 15% of data missing we discard that variable. In this case it will be PoolQC, MiscFeature, Alley, Fence, FireplaceQu, LotFrontage. If judge the variables they are not key metreics for buying a house. They might add on to the cost of the house but how ?
# * Garage X varbales have the same number of missing data. After checking they seem to be the same observartions. We already are including Garagecars variable which contains important information, we can go ahead and delete them too. Same for Bsmt variables
# * For MassVnr variables did not show a strong co-relation to our target variable and are strongly correlated to OverallQual and YearBuilt which are already taken into account.
# * For  Electrical it is just one row and we can delete that observation

# In[ ]:


# deleting with missing data
train_df = train_df.drop(missing_data[missing_data['Total'] > 1].index,1)
train_df = train_df.drop(train_df.loc[train_df['Electrical'].isnull()].index,)
train_df.isnull().sum().max() 


# **Now we handle outliers**
# * Outliers can pull  the model in one direction too much and it is important to find them and find out ways to reduce their impact
# * To begin we have to establish a threshold that will help us identify outliers. One way is to standardize the data
# * Data standardization  means converting data values to have mean of 0 and a standard deviation of 1

# In[ ]:


# Univariate analysis of Sale Price
sale_price_scaled = StandardScaler().fit_transform(train_df['SalePrice'][:,np.newaxis])
low_range = sale_price_scaled[sale_price_scaled[:,0].argsort()][:10]
high_range = sale_price_scaled[sale_price_scaled[:,0].argsort()][-10:]
print('Outer range (low) of the distribution: ')
print(low_range)
print('Outer range (high) of the distributions: ')
print(high_range)


# * Low range values are similar and not too far from 0
# * High range values are far   from 0 and the 7 something values are realy out of range

# **Bivariate analysis**
#  :We will analyze our scatter plots again to look at outliers this time[](http://)

# In[ ]:


var = 'GrLivArea'
data = pd.concat([train_df['SalePrice'], train_df[var]],axis=1)
data.plot.scatter(x=var, y='SalePrice',ylim=(0,800000))


# * The two values with bigger area do not follow the crowd. It could be a different type of property
# * The two values above 7000000+ are exponentially higher than other values. It still follows a linear relationship so we will keep it

# In[ ]:


# deleting outlier points
train_df.sort_values(by='GrLivArea', ascending = False)[:2]


# In[ ]:


train_df = train_df.drop(train_df[train_df['Id']==1299].index)
train_df = train_df.drop(train_df[train_df['Id']==524].index)


# In[ ]:


var = 'TotalBsmtSF'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
data.plot.scatter(x=var,y='SalePrice', ylim=(0,800000))


# * We do a lot of points that see   like  outliers. The observations where 3000 BsmtSF is more that 3000 but it does not follow linear relationship
# * The observations that have 0 BsmtSF in real world will still have value as the value of the house is observed by living  area and other amenities

# In[ ]:


#deleting outliers
train_df.sort_values(by='TotalBsmtSF',ascending = False)[:2]


# In[ ]:


train_df.drop(train_df[train_df['Id']==333].index, inplace=True)
train_df.drop(train_df[train_df['Id']==497].index, inplace=True)


# **Let's try some mulitvariate analysis**
# * But before we go ahead with multivariate analysis we should validate that does 'SalePrice' compliy with statistical assumptions.

# The follwoing four assumptions should be tested:
# * Normality - Data should look like a normal distribution. As part of this we will check the  univariate normality for 'SalePrice'. One details to take into account is that in big samples(>200 observations) normality is not such an issue.
# * Homoscedasticity - refers to the assumption that dependent variables exhibit equal levels of variance across the range of predictor variables
# * Linearity - The most common way to assess linearity is to examine scatter plots and search for linear pattersn. If patterns are not linear, data transformation can be used as an avenue.
# * Absence of correlated erros - happen when one error is correlated to another. For example, if one positive error makes a negative error systematically, it means that there is a relationship between these variables.
# 

# Let's test the normality of Sale Price using
# - Histrogram: Kurtosis and skewness
# - Normal probability plot: Data distribution should closely follow the diagonal that represents the normal distribution

# In[ ]:


#histogram and normal probablity plot
sns.distplot(train_df['SalePrice'], fit=norm)
fig=plt.figure()
res=probplot(train_df['SalePrice'], plot=plt)


# It is evident 'Sale Price' is not normaly distributed. Demonstrate positive skewness and does not follow the diagonal line
# We can improve this behaviour through data transformation. In case of positive skewness, log transformations usually works well.

# In[ ]:


# apply log transformations
train_df['SalePrice'] = np.log(train_df['SalePrice'])


# In[ ]:


#transformed histogram and normal probability plot 
sns.distplot(train_df['SalePrice'], fit=norm)
fig = plt.figure()
res = probplot(train_df['SalePrice'], plot=plt)


# Let's analyze GrLivArea

# In[ ]:


sns.distplot(train_df['GrLivArea'], fit=norm)
fig = plt.figure()
res = probplot(train_df['GrLivArea'], plot=plt)


# We see a slight positive skewness

# In[ ]:


#Living area transformation
train_df['GrLivArea'] = np.log(train_df['GrLivArea'])


# In[ ]:


sns.distplot(train_df['GrLivArea'], fit=norm)
fig = plt.figure()
res = probplot(train_df['GrLivArea'], plot=plt)


# Time to look at Basement square feet

# In[ ]:


sns.distplot(train_df['TotalBsmtSF'], fit=norm)
fig = plt.figure()
res = probplot(train_df['TotalBsmtSF'], plot = plt)


# In[ ]:


# Create a categorical variabe to identify if there is a basement
train_df["HasBsmt"] = pd.Series(len(train_df["TotalBsmtSF"]), index=train_df.index)
train_df['HasBsmt'] = 0
train_df.loc[train_df['TotalBsmtSF']>0,'HasBsmt'] = 1


# In[ ]:


# Transform data
train_df.loc[train_df['HasBsmt']==1,'TotalBsmtSF'] = np.log(train_df['TotalBsmtSF'])


# In[ ]:


#histogram and normal probability plot
sns.distplot(train_df[train_df['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm)
fig = plt.figure()
res = probplot(train_df[train_df['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)


# One of the best way to approach homoscedasticity for two metice variables is graphically. Departures from an equal dispersion are shown  by shapes as cone(small depression at one side of the graph, large dispersion at opposite side) or diamonds( a large number of points at the center of the distribution)

# In[ ]:


# Scatter plot between living area and sale price after  transformation
plt.scatter(train_df['GrLivArea'], train_df['SalePrice'])


# Previous to log transformation this scatter plot had a cone shape. As you can  see, the current scatter plot doesn't have a conice shape anymore. 

# In[ ]:


# Scatter plot between Basement area and Sales price
plt.scatter(train_df[train_df['TotalBsmtSF']>0]['TotalBsmtSF'], train_df[train_df['TotalBsmtSF']>0]['SalePrice'])


# Don't forget to convert categorical variables into dummy variables before feeding them to the model

# In[ ]:


train_df = pd.get_dummies(train_df)


# In[ ]:


train_df.dtypes


# This kernel is practice kernel and followed [https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python](http://)
# Thank you very much for educating all of us

# In[ ]:




