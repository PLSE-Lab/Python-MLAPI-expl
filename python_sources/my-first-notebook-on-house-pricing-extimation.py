#!/usr/bin/env python
# coding: utf-8

# Hello Everybody,
# This is my first notebook here on Kaggle and I am trying to familiarize with this interface together with getting more experience on the handling of typical ML problems.
# 
# Below I've collected some script examples from other kernels and I would like to perform the following steps:
# 
#  1. Dataset analysis and visualization 
#  2. Feature scaling and mean normalization 
#  3. Evaluate several model to choose the best one by calculating the "learning curves" of each one
#  4. Let's try to address  over/under-fitting using test and train data set properly. 
#  5. Apply     regularization if needed in the proper way. 
# 
# Finally I will try to understand what to do next by looking to the previous results..

# In[ ]:


# Python script for test
#taking inspiration from Poonam Lingade kernel on data visualization:
#https://www.kaggle.com/poonaml/house-prices-advanced-regression-techniques/house-prices-data-exploration-and-visualisation

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="whitegrid", color_codes=True)
sns.set(font_scale=1)
houses=pd.read_csv("../input/train.csv")
houses.head()


# In[ ]:


hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))


# In[ ]:


#lets see if there are any columns with missing values 
null_columns=houses.columns[houses.isnull().any()]
houses[null_columns].isnull().sum()


# In[ ]:


#descriptive statistics summary
houses['SalePrice'].describe()


# In[ ]:


#histogram
sns.distplot(houses['SalePrice']);


# In[ ]:


#skewness (asimmetria) and kurtosis
print("Skewness: %f" % houses['SalePrice'].skew())
print("Kurtosis: %f" % houses['SalePrice'].kurt())


# In[ ]:


#scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([houses['SalePrice'], houses[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[ ]:



#Compute pairwise correlation of columns, excluding NA/null values
corrmat = houses.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True);


# In[ ]:


#saleprice correlation matrix
k = 10 #number of variables for heatmap
#Get the rows of a DataFrame sorted by the k largest values of 'SalePrice' column
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
#Return Pearson product-moment correlation coefficients: Ri,j = Ci,j /(sqrt(Ci,i*Ci,j))
cm = np.corrcoef(houses[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
#plt.show()


# '**GarageCars**' and '**GarageArea**' are some of the most strongly correlated variables. However, as we discussed in the last sub-point, the number of cars that fit into the garage is a consequence of the garage area. 'GarageCars' and 'GarageArea' are like twin brothers. You'll never be able to distinguish them. Therefore, we just need one of these variables in our analysis.
# we can keep '**GarageCars**' since its correlation with '**SalePrice**' is *higher*!!

# In[ ]:


#missing data
total = houses.isnull().sum().sort_values(ascending=False)
#check = houses.isnull().count() #this gives the number of elements for each column (feature)

percent = (houses.isnull().sum()/houses.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[ ]:




#missing data
total = houses.isnull().sum().sort_values(ascending=False)
percent = (houses.isnull().sum()/houses.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[ ]:


#dealing with missing data
df_train = houses.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train.isnull().sum().max() #just checking that there's no missing data missing...
#df_train['Electrical'].count() #we've remevode the NA element from electrical
#now df_train has less features than in the beginning!


# Let's **standardize features** by removing the mean and scaling to unit variance
# From scikit definition:
# 
# Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set. Mean and standard deviation are then stored to be used on later data using the transform method.
# 
# Standardization of a dataset is a common requirement for many machine learning estimators: they might behave badly if the individual feature do not more or less look like standard normally distributed data (e.g. Gaussian with 0 mean and unit variance).
# 
# For instance many elements used in the objective function of a learning algorithm (such as the RBF kernel of Support Vector Machines or the L1 and L2 regularizes of linear models) assume that all features are centered around 0 and have variance in the same order. **If a feature has a variance that is orders of magnitude larger that others, it might dominate the objective function** and make the estimator unable to learn from other features correctly as expected.

# In[ ]:


#Fits transformer to X and y with optional parameters fit_params 
#returns a transformed version of X.
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)


# Bivariate Analysis

# In[ ]:


var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[ ]:




#sort the df_train values for GrLivArea then we take the first 2 elements
#that are our outliers (they don't follow the crew)
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]


# In[ ]:


#drop ID 1299 and 524 away from our dataset
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

#plot again without the outliers
#bivariate analysis saleprice/grlivarea
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[ ]:




#histogram and normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()#create a new figure for the probability plot:
res = stats.probplot(df_train['SalePrice'], plot=plt)


# "In case of positive skewness, log transformations usually works well"
# Also from prof. Ng lessons on "choosing what feature to use " (week 9 of Machine Learning course) there is an explanation where he stated that "non- Gaussian" features are more likely to be a Gaussian if they are "log" filtered., e.g. x1 --> log(x1)

# In[ ]:


#let's evaluate the sqrt transformation
df_train['SalePrice'] = np.sqrt(df_train['SalePrice'])
#transformed histogram and normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)


# In[ ]:


#applying log transformation
df_train['SalePrice'] = np.log(df_train['SalePrice'])
#transformed histogram and normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)


# Let's do the same also for the other main features

# In[ ]:


#histogram and normal probability plot
sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)


# In[ ]:


Log_1stFlrSF =np.log(df_train['1stFlrSF'])
#transformed histogram and normal probability plot
sns.distplot(Log_1stFlrSF, fit=norm);
fig = plt.figure()
res = stats.probplot(Log_1stFlrSF, plot=plt)


# In[ ]:


#let's create a new feature by multiplying the log(GrLivArea*1stFlrSF) 
Log_newFeature = np.log(df_train['GrLivArea']*df_train['1stFlrSF'])

#transformed histogram and normal probability plot
sns.distplot(Log_newFeature, fit=norm);
fig = plt.figure()
res = stats.probplot(Log_newFeature, plot=plt)


# Let's check the scatter  plot of this **new feature**:

# In[ ]:



plt.scatter(Log_newFeature, df_train['SalePrice']);


# In[ ]:


#histogram and normal probability plot
sns.distplot(df_train['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)


# In[ ]:




#histogram and normal probability plot
sns.distplot(df_train['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)


# About this last feature, we have many zeros value then we cannot apply log transformation.

# In[ ]:




#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0 
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1


#transform data
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])


#histogram and normal probability plot
sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)


# The best approach to test homoscedasticity for two metric variables is graphically. Departures from an equal dispersion are shown by such shapes as cones (small dispersion at one side of the graph, large dispersion at the opposite side) or diamonds (a large number of points at the center of the distribution).

# In[ ]:


#scatter plot
plt.scatter(df_train['GrLivArea'], df_train['SalePrice']);


# In[ ]:




#scatter plot
plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice']);


# In[ ]:




