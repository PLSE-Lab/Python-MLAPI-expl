#!/usr/bin/env python
# coding: utf-8

# # **EDA for beginners with in depth explaination!!!**

# I read a lot of kaggle kernels online but noticed that very less people are explaining things properly so I decided to make a kernel which explains the thought process required when performing an EDA(Exploratory Data Analysis)

# ## *Load dataset from the kaggle API*

# In[ ]:


import pandas as pd
train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# # **Importing the necessary libraries**

# In[ ]:


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


# # **Step 1: Understanding the Data**

# First let us take a peek at our data!!!

# In[ ]:


train.head()


# Lets have a look at the Dimensions of the data

# In[ ]:


train.shape


# We notice that this is a small dataset.
# Good news: algorithms take less time to train.
# Bad News: We might not have enough data to train the algorithms. Also we observe the  dataset has a lot of features.
# Lot of features means eliminating noisy features and also the curse of dimensionality comes into play here.

# In[ ]:


print(train.dtypes.unique())


# We have 3 types of data in our dataset- Integer,Float and Object. This information is good to know for data cleaning

# In[ ]:


len(train.select_dtypes(include=['O']).columns)


# In[ ]:


len(train.select_dtypes(include=['int64']).columns)


# In[ ]:


len(train.select_dtypes(include=['float64']).columns)


# We have 46 numerical features and 35 categorical/ordinal features 

# # **Descriptive Statistics**

# The describe() function of pandas gives us 8 statistical values - count,mean,standard deviation,minimum value, 25th percentile, 50th percentile,75th percentile, maximum value

# Missing values can be detected using count

# The other 7 statistical values can be used for detecting outliers

# In[ ]:


pd.set_option('precision', 3)
train.describe()


# In[ ]:


y=train.count()[train.count()!=1460.000]


# In[ ]:


y.shape


# There are about 19 missing values

# In[ ]:


sns.barplot(y.index,y.values)
plt.xticks(rotation=90)
plt.show()


# We observe that most of the columns have very less missing values.
# Missing values substitution will be critical for PoolQC,Fence,MiscFeature,Alley,LotFrontage,FireplaceQu

# # **Correlations between Attributes**

# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(train.corr(method='pearson'))


# We observe that SalePrice is directly correlated to OverallQual,LotFrontage,YearBuilt,YearRemodAdd,MasVnrArea,TotslBsmtSF,1stFlrSF,GrLivArea,FullBath,Garage Cars, Garage Area.

# # **Skew of Univariate Distributions**

# Important because many machine learning algorithms assume a Guassian distribution which has zero skew

# In[ ]:


y=abs(train.skew())
plt.figure(figsize=(7,7))
sns.barplot(y.index,y.values)
plt.xticks(rotation=90)
plt.show()


# We can observe that only 5 features have a major skew rest all features have a skew of less than 5.00

# # **Step 2: Data Visualisation**

# # **Univariate Plots**

# Using histograms we can easily check the skew for each variable

# In[ ]:


train.hist(figsize=(20,20))
plt.show()


# Distplots are histplots with curves on them

# In[ ]:


x=train.select_dtypes(exclude='object')
f=train.describe().columns
y=train.count()[train.count()==1460].index
g=f.intersection(y)
for feat in g:
  sns.distplot(train[feat])
  plt.show()


# Box and whisker plots for detecting outliers

# In[ ]:


g=train.select_dtypes(exclude='object').columns
for x in g:
  sns.boxplot(train[x],orient='v')
  plt.show()


# # **Multivariate Plots**

# # **Correlation Plot**

# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(train.corr(method='pearson'))


# # **Scatter Matrix Plot**

# In[ ]:


# since scatter_matrix takes a lot of time taking less features for demo
g=['SalePrice','LotArea','OverallQual','YearBuilt','YearRemodAdd','MasVnrArea','TotalBsmtSF','GarageCars','OverallCond','KitchenAbvGr']
sns.pairplot(train[g])


# # **Summary**

# 5 key things to look for in a dataset

# 1. Normality 
# 
# We do this with the help of skew plots,distplots and histograms
# 
# 

# 2. Linearity
# 
# We do this with the help of correlation matrix,scatter_matrix and pairplots

# 3. Missing values and Outliers
# 
# We do this with the help of boxplots,describe() and count of missing values.

# 4. Feature cleaning
# 
# Checking data types help us to know what type of features have to be cleaned.

# 5. Overfitting, Feature selection and Data Augmentation
# 
# Knowing the dimensions of the dataset helps us to deal with these issues 
