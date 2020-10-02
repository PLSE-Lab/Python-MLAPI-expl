#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis

# Dataset used is Advanced Housing Price dataset of Kaggle.
# **The aim of this EDA is to understand the basic and necessary steps followed**. The steps might differ but the approach mostly remains constant. 
# 
# When performing EDA we necessarily perform some of the steps as:
#     1. Loading the data
#     2. Identifying the dependent and target features (as per the problem)
#     3. Identifying the categorical and continuous features (we need to apply different rules)
#     4. Finding null values and dealing with them
#     5. Getting familiar with the various types of data present in the dataset
#     6. Check the skewness/distribution of the data and outliers
#     7. Feature engineering
#     8. Correlation
#     9. Understanding the contribution of non-numeric features on target variable
#     
# In advanced versions of EDA certain tests such as Chi-Square and Anova test are applied to get better insight into the distribution of data (At present we need not worry about these tests)

# **A small note about chi-square and anova tests for more curious readers:**
# Chi-square is a mathematical relation which determines the relation between nominal attributes. It takes into account the frequency measurements of joint occurence of two attributes.
# 
# Anova test is used to measure the separation of data samples from one another. It is the measurement of difference between samples of data.

# In[1]:


import os
import pandas as pd


# In[2]:


os.listdir("../input")


# In[3]:


data = pd.read_csv(os.path.join("../input","train.csv"))


# In[4]:


data.shape       #shape of the training set


# In[6]:


data.head(5)


# In[7]:


data.columns


# 81 features including target feature('SalePrice') and 'Id'

# In[ ]:


data = data.drop(columns = ['Id'])


# SalePrice is our target variable while all others are predictor variables

# In[ ]:


data.info()


# It can be seen that some of the features in the dataset have null values

# ### Missing Values in the data

# In[ ]:


data.isnull().sum()


# We can see the number of null values in each of the features

# The categorical and continuous features need to be explored differently. Although it may apparently seem that the numerical columns are continuous while the non-numeric columns are categorical, this is not true. For example, the feature 'OverallQual' is of dtype('int64') it is an ordinal feature(categorical).Thus all the non-numeric features are categorical but the reverse is not true.
# In this Analysis numerical and non-numeric features are analysed separately

# In[ ]:


num_features = data._get_numeric_data().columns  #numeric features may or may not be categorical
cat_features = list(set(data.columns) - set(num_features))


# In[ ]:


len(num_features)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


sns.kdeplot(data['SalePrice'])


# In[ ]:


plt.figure(figsize = (50,40))
for i in range(1,len(num_features)):
    plt.subplot(8,5,i)
    plt.title(num_features[i])
    sns.kdeplot(data[num_features[i]])


# The visualisation depicts the data distribution of the numerical features in terms of data density. The data is not normally distributed
# 

# ### Presence of outliers in the data 

# In[ ]:


data.describe()


# The description points out the presence of outliers in most of the numerical features as the gap between the 75th % tile and max is large. There are also outliers in the other extreme where the gap between the minimum and 25th %tile is large. We plot box-plots to make it clear with visualisations

# In[ ]:


plt.figure(figsize = (80,30))
for i in range(1,len(num_features)):
    plt.subplot(5,8,i)
    plt.title(num_features[i])
    sns.boxplot(data[num_features[i]], orient = "v")


# In[ ]:


data[['YearBuilt','YearRemodAdd','YrSold','SalePrice']].sample(10).sort_values(by=['YearBuilt'])


# In[ ]:


x = data.loc[data['YearBuilt']>=2000,['YearBuilt']].sort_index(ascending = False)
x.count()   #number of houses built between 2000 and 2010 out of 1460


# In[ ]:


data['YearBuilt'].unique()


# In[ ]:


data['YearBuilt'].value_counts().sort_index().plot()


# Most of the houses in our data set are built around 2000 and later

# In[ ]:


data.loc[data['YearBuilt']>=2000,['YearBuilt','YrSold','SalePrice']].sample(10)


# In[ ]:


d = data[['YrSold','SalePrice']].sort_values(by=['YrSold'])
d.sample(5)


# In[ ]:


d['YrSold'].unique()


# In[ ]:


d1 = d.groupby(['YrSold'])
d1.describe()


# The data provides an insight into the reduction in SalePrice around year 2008

# In[ ]:


plt.plot(d1.mean())
plt.title('Mean Sale Price vs Year Sold')
plt.xlabel('YrSold')
plt.ylabel('MeanSalePrice')


# The visualization provides certainity to the hypothesis that SalePrice drops around year 2008 and is high around year 2007

# In[ ]:


m = data[['MoSold','SalePrice']].sort_values(by=['MoSold']).groupby(['MoSold'])
m.describe()


# Most of the houses are sold in the 5th, 6th and 7th months

# In[ ]:


plt.plot(m.mean())
plt.title('Mean Sale Price vs Month Sold')
plt.xlabel('Month Sold')
plt.ylabel('Mean Sale Price')


# There is comparatively some variation in the price based on the month in which the house is sold. The peak price could be a consequence of demand of houses in a particular season

# ### Feature Engineering

# In[ ]:


b = data[['BedroomAbvGr','1stFlrSF','2ndFlrSF','SalePrice']].sort_values(by=['BedroomAbvGr'])
b['SF_per_bed'] = (b['1stFlrSF']+b['2ndFlrSF'])/b['BedroomAbvGr']
b.groupby(['BedroomAbvGr']).describe()


# In[ ]:


x = b.groupby(['BedroomAbvGr'])
x = x.mean()[x.count()>50]     #excluding some of the outliers
x = x.loc[x['SalePrice']>0]
x


# In[ ]:


plt.plot(x['SalePrice'])
plt.xlabel('BedroomAbvGr')
plt.ylabel('MeanSalePrice')


# Mean Housing prices increase with an increase in number of bedrooms

# In[ ]:


b1 = x.sort_values(by=['SF_per_bed'])
b1


# In[ ]:


data[['ScreenPorch','SalePrice']].sample(12)


# In[ ]:


data['ScreenPorch'].unique()


# In[ ]:


data['ScreenPorch'].value_counts().head()


# Most of the houses don't have screen porch (1344 out of 1460)

# In[ ]:


data_life = data.loc[:,['YearRemodAdd','SalePrice']]    
    #num_of_years is the age of the house at the time of selling
data_life['num_of_years'] = data['YrSold'] - data['YearBuilt']
data_life.sample(7)


# In[ ]:


dl = data_life[['num_of_years','SalePrice']].sort_values(by=['num_of_years']).groupby(['num_of_years'])
dl.describe().sample(5)


# In[ ]:


x = dl.mean()[dl.count()>5]
x = x.loc[x['SalePrice']>0]
plt.plot(x)
plt.title('Mean Sale Price vs Age of House')
plt.xlabel('AgeofHouse')
plt.ylabel('MeanSalePrice')


# on excluding values which have count<=5 (outliers) it is certain that the mean Sale Price drops with the increasing num_of_years

# In[ ]:


corr = data.corr()
corr


# In[ ]:


plt.figure(figsize = (32,30))
sns.heatmap(corr,cmap = 'nipy_spectral', annot = True)


# The representation shows that most of the features are not linearly related to SalePrice and if we train a model using linear regression we may even drop features having correlation ~0 with respect to SalePrice. For features like 'GarageCars' and 'GarageArea', the correlation among them is high which is intuitive. Only one of such features is sufficient for the model and the other may be dropped if linear regression is to be used.

# 'OverQual' is having high linear correlation with the 'SalePrice'. It can be said with certainity that high value of 
# 'OverQual' will have relatively high 'SalePrice' and vice-versa

# In[ ]:


x = data.loc[:,['OverallQual','SalePrice']].groupby(['OverallQual'])
x.describe()


# In[ ]:


x = x.mean()[x.count()>5]
x = x.loc[x['SalePrice']>0]


# In[ ]:


plt.plot(x)
plt.title('MeanSalePrice vs OverallQual')
plt.xlabel('Overall Quality')
plt.ylabel('MeanSalePrice')


# The mean sale price of the house increases with the increase in overall quality

# In[ ]:


plt.figure(figsize = (50,40))
for i in range(0,len(cat_features)):
    plt.subplot(8,6,i+1)
    #plt.title(cat_features[i])
    sns.countplot(data[cat_features[i]])

#double click images to zoom


# Some of the features contain data of only one category and negligible in others. They contribute comparitively less in predicting the price of a house.Features like 'Neighbourhood', 'Foundation', 'KitchenQual' intuitively contribute significantly in determining 'SalePrice'

# Features having values of one type are described. There are many such features, here 3 are described

# In[ ]:


data['Utilities'].describe()


# In[ ]:


data['RoofMatl'].describe()


# In[ ]:


data['Heating'].describe()


# ### Analysing contribution of non-numeric features on SalePrice 

# In[ ]:


data_above_mean_sp = data.loc[data['SalePrice']>=181000]
data_below_mean_sp = data.loc[data['SalePrice']<181000]


# In[ ]:


data_above_mean_sp.shape


# In[ ]:


data_below_mean_sp.shape


# In[ ]:


int_col = ['Neighborhood','Foundation','Exterior2nd','HouseStyle','KitchenQual']


# In[ ]:


plt.figure(figsize = (25,50))
i = 0
print('\tSale Price above 181000\t\t\t\t\t\tSale Price below 181000')
for col in int_col:
    i = i + 1
    plt.subplot(5,2,i)
    label_am = data_above_mean_sp[col].unique()
    size_am = data_above_mean_sp[col].value_counts()
    #print(size_am, label_am)
    plt.pie(size_am,labels = label_am, autopct = '%1.1f%%',shadow = True,startangle = 0)
    plt.legend()
    plt.title(col)
    i = i + 1
    plt.subplot(5,2,i)
    label_bm = data_below_mean_sp[col].unique()
    size_bm = data_below_mean_sp[col].value_counts()
    #print(size_bm)
    plt.pie(size_bm, labels = label_bm,autopct = '%1.1f%%', shadow = True, startangle = 0)
    plt.legend()
    plt.title(col)


# From the visualisation, a house having 1 story will have price above 181000 with high probability. The same goes with other features such as 'Foundation'. 
# A house 'PConc' as its foundation will have a price above 181000 with very high probability whereas one having 'Brktil' will have lower price with some certainity.
# This provides some certainity to analyse the 'SalePrice'

# In[ ]:


x = data[['KitchenQual','KitchenAbvGr','SalePrice']].sort_values(['SalePrice'])
x.sample(8)


# In[ ]:


data['KitchenQual'].value_counts()


# In[ ]:


data['KitchenAbvGr'].value_counts()


# Most of the houses have only one kitchen (1392 out of 1460)

# In[ ]:


x = x.drop(['KitchenAbvGr'], axis = 1)


# In[ ]:


x = x.groupby(['KitchenQual'])
x.describe()


# Ex: Excellent, Gd: Good, TA: Typical/Average Fa: Fair
# Thus it is certain that a kitchen having better quality with lead to high SalePrice

# In[ ]:


plt.plot(x.mean()['SalePrice'])


# The mean Sale Price increases with better kitchen quality

# In[ ]:


data['MSZoning'].value_counts()


# Majority of the houses in the given data are in Residential Low Density area

# In[ ]:


x = data.loc[data['GrLivArea']<3000,['GrLivArea','SalePrice']].sort_values('GrLivArea')
x.sample(6)


# The Sale Price varies with increase in Ground Living Area

# In[ ]:


sns.scatterplot('GrLivArea','SalePrice',data = x)


# In[ ]:


data[['OverallCond','SalePrice']].sample(5)


# The EDA thus provides an insight into the data and helps to understand the pattern, outliers and prominent features in the
# dataset

# This is one of my starting works. If you like my work, and was able to gain a small insight, please upvote and help me improve with your suggestions.

# In[ ]:




