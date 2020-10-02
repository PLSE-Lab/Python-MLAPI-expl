#!/usr/bin/env python
# coding: utf-8

# # <center>House Price</center><br>
# <img src='http://clipart-library.com/images_k/houses-silhouette/houses-silhouette-5.png'>

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np


# In[ ]:


data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
data.head()


# # Data exploration and cleaning

# In[ ]:


data.shape


# In[ ]:


data.columns


# In[ ]:


data.dtypes


# In[ ]:


data.isnull().sum().sort_values(ascending=False)[:20]


# After reading the description.txt, I choose to delete columns that has no interest in my research or are in duplicates

# In[ ]:


data = data.drop(columns=['Id','Street'],axis=1)


# I also delete the column PoolQC which has a lot of missing values

# In[ ]:


data = data.drop(columns=['PoolQC','MiscFeature'],axis=1)


# Adding a new feature TotalBathrooms with sum of FullBath and HalfBath + in basement
# 

# In[ ]:


data['TotalBathrooms'] = (data['FullBath'] + (0.5 * data['HalfBath']) + data['BsmtFullBath'] + (0.5 * data['BsmtHalfBath']))


# When NA means 'none' in the description.txt I fill the missing values of the features by 'None'

# In[ ]:


for col in ('FireplaceQu',
            'Fence',
            'Alley',
            'GarageType', 
            'GarageFinish', 
            'GarageQual', 
            'GarageCond'):
    data[col]=data[col].fillna('None')


# After checking if the house has a garage, I fill the missing values of GarageYrBlt by the median value. 
# If there is no garage I fill GarageYrBlt by 0

# In[ ]:


data.loc[data['GarageType']!='None', "GarageYrBlt"] = data["GarageYrBlt"].fillna(data['GarageYrBlt'].median())
data["GarageYrBlt"]=data["GarageYrBlt"].fillna(0)


# I fill the missing values of the column LotFrontage by the median value

# In[ ]:


data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))


# Making false numerical variables into categorical

# In[ ]:


data['YrSold'] = data['YrSold'].astype(str)
data['OverallCond'] = data['OverallCond'].astype(str)
data['MoSold'] = data['MoSold'].astype(str)
data['MSSubClass'] = data['MSSubClass'].apply(str)


# Encoding the dataframe into another one

# In[ ]:


encoded_data = pd.get_dummies(data)
encoded_data.head()


# # Data research

# Study of the Base variable = SalePrice

# In[ ]:


data['SalePrice'].describe()


# Displaying distribution of the target variable

# In[ ]:


data['SalePrice'].hist(bins=50)


# Adjusting skewed data distribution

# In[ ]:


encoded_data['SalePrice_skewed'] = np.log1p(data['SalePrice']) 
encoded_data['SalePrice_skewed'].hist(bins=50)


# Displyaing top 10 of data correlated to SalePrice

# In[ ]:


encoded_data[encoded_data.columns[1:]].corr()['SalePrice_skewed'][:].sort_values(ascending=False)[2:12]


# # Statistical Analysis

# I choose to study the Ground Live Area vs Sale Price. There is a linear regression

# In[ ]:


plt.figure(figsize=(20,8))
sns.regplot(x='GrLivArea', y="SalePrice_skewed", data=encoded_data, color='green')


# Checking the reasons of outlier : GrLivArea > 4000

# In[ ]:


encoded_data[(encoded_data['GrLivArea']> 4000) & (encoded_data['SalePrice_skewed']<13)]


# The reason is that the SaleCondition = Partial (means Home was not completed when last assessed)
# I delete those rows.

# In[ ]:


encoded_data.drop([523,1298], inplace=True)


# I choose to study the impact of OverallQual on Sale Price.

# In[ ]:


plt.figure(figsize=(10,8))
sns.boxplot(x="OverallQual", y="SalePrice_skewed", data=encoded_data)


# I choose to study the Year of construction vs Sale Price. There is a linear regression 

# In[ ]:


plt.figure(figsize=(20,8))
sns.regplot(x='YearBuilt', y="SalePrice_skewed", data=encoded_data)


# I choose to study the existence and surface of the basement vs Sale Price. There is a linear regression 

# In[ ]:


plt.figure(figsize=(20,10))
sns.regplot(x='TotalBsmtSF', y="SalePrice_skewed", data=encoded_data, color='purple')


# Impact of Full Bathrooms on Sale Price :

# In[ ]:


plt.figure(figsize=(10,8))
sns.boxplot(x="TotalBathrooms", y="SalePrice_skewed", data=encoded_data)


# Checking why there are outliers when +5 bathrooms

# In[ ]:


from math import exp
pd.set_option('display.max_columns', 500)
outliers = data[(data['TotalBathrooms']>= 5) & (data['SalePrice']<exp(13))]
outliers


# <br>The BsmtFinType2 indicates Unfinished, this could explain the low sale price. Moreover, the values in YearBuilt are below 1990 : the houses are old
# <br>Finally, the OverallQual values are medium (5/10)

# I choose to study the existence and surface of a first floor vs Sale Price. There is a linear regression 

# In[ ]:


plt.figure(figsize=(20,10))
sns.regplot(x='1stFlrSF', y="SalePrice_skewed", data=encoded_data, color='orange')


# Checking R-squared with the 4 most correlated features

# In[ ]:


X = sm.add_constant(encoded_data[['GrLivArea','TotalBathrooms','OverallQual','GarageCars']])
Y = encoded_data['SalePrice_skewed']

model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 

print_model = model.summary()
print(print_model)


# The model explains 82% of the variance of Sale Price !

# Even if the correlation coefficient didn't show it, I presumed that the neighborhood had an impact on the Sale Price.

# In[ ]:


plt.figure(figsize=(30,10))
sns.boxplot(x=data['Neighborhood'], y=data['SalePrice'], data=data)


# Using ANOVA to test H0:the mean SalePrice of every neighborhoods are equals

# In[ ]:


model = ols('SalePrice ~ Neighborhood', data = data).fit()
                
anova_result = sm.stats.anova_lm(model, typ=2)
print (anova_result)


# H0 is rejected : the means are not equal, there is a difference of SalePrice between neighborhoods
# But based on our correlation result, it is not the main features to influence the SalePrice

# In[ ]:




