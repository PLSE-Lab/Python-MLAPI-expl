#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Outline
# 1. Problem description
# 1. Import libraries 
# 1. Upload Datasets
# 1. Understanding the Data
# 1. Data Cleaning
# 1. feature Engineering
# 1. Train The Model
# 
# 

# # **1. Problem Description**
# 
# We need to predict the final price of a residential homes in Ames, Iowa given its attributes.
# 
# People buy houses to satisfy a certain need or a set of needs. It could be a combination of space for kids, procimity to schools, prestige etc.  The more people are looking for a certain feature to satisfy their need the more the demand, that pushes the price to go high and vice versa. For every house that was sold we want to determine how important each feature was and how much did it impact the price.

# # 2. Import libraries
# 

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas_profiling


# # 3. Upload The Datasets

# In[ ]:


train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# # 4. Understanding our Data

# In[ ]:


train.head(3)


# In[ ]:


test.head(3)


# In[ ]:


train.shape, test.shape


# In[ ]:


train.info()


# In[ ]:


train.describe()


# In[ ]:


# lets understand our target variable sale price
# descrptive stats of price

train ['SalePrice'].describe()


# No missing price, max 755,000 min 43,900 In real life we can say that its a great difference
# 

# In[ ]:


sns.distplot(train['SalePrice']);


# The price is skewed to the left, with most home prices being btwn 50000 and 300,000, 500,000 and above are replecented by a smaller population.

# In[ ]:


# what is the relationship between the price and the rest of the features
# Is there a correlation
# Perform correlation

corr = train.corr()
corr.sort_values(["SalePrice"], ascending = False, inplace = True)
print(corr.SalePrice)


# Some of of the top corr features: OverallQual, GrLivArea, GarageCars, GarageArea, TotalBsmtSF, 1stFlrSF, FullBath, TotRmsAbvGrd, YearBuilt, YearRemodAdd Least corr includes: YrSold
# 
# Lets do a scatter plot of the features most corr to price for a better fiew

# In[ ]:


#scatterplot of the above identified top corr features

sns.set()
cols = [ 'SalePrice','OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'TotRmsAbvGrd', 'YearBuilt']
sns.pairplot(train[cols], height = 2.5)
plt.show();

# after viewing the scatter plots, I have adjusted the code to remove FullBath and GarageCars - they seem not to give much info


# We can confirm the strong correlation. Some of what we are observing is common knowledge, what is interesting is the detail of the relationship
# The price increase with newer houses, we also see increase as features increase upto to the highest value such as the TotalBsmtSF

# In[ ]:





# # 5. Data Cleaning
# * Are there any irregularities in this data set and how do we deal with it?
# * Is there missing data and how do we deal with it?
# * Do we have any duplicate entries that we need to delete?
# * Do we have columns that will be irrelevant to our analysis that we may need to delete such as Id?
# * Do we have outliers that can affect our analysis, and how do we deal with it?

# In[ ]:


# We have identified missing values in various columns in our previous codes
# Lets see missing values in an assending order
# adjust count to only show values with missing data
# are the features among the most corr or least corr

total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/1460).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# Lets reveiw all the variables with missing data
# 
# We should consider dropping variables with more than 50% missing values. 
# Looking at the first 5 variables with close to 50% and above,and compare it with the corr table we can conclude that these varibles are of no much value to our analysis : PoolQC,MiscFeature, Alley, Fence, FireplaceQu. Hence decision to delete.
# 
# The garage variables have the same # of missing data, possibly the same observations. .
# 
# For the rest of the variables lets replace with a median apart from electrical which has only one missing observation which we are going to delete

# In[ ]:


# drop the null observation in Electrical 
train = train.drop(train.loc[train['Electrical'].isnull()].index)


# In[ ]:


# drop all columns with null
# Note for better performance, we will review this section

train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)


# In[ ]:


train.shape


# In[ ]:


# 2. check for duplicates entries, in Id
# If duplicates. . delete
# drop Id column since we will not need to use it in our analysis

id_unique = len(set(train.Id))
id_total = train.shape[0]
id_dup = id_total - id_unique
print(id_dup)

#drop id column
train.drop(['Id'],axis =1,inplace=True)

# no duplicates :-)


# In[ ]:


# check the shape of the train dataset at this point
train.shape


# In[ ]:


# Outliers
# data values 1.5 times the interquartile range above the third quartile or below the first quartile - IQR rule

# outliers in price


# In[ ]:


stat = train.SalePrice.describe()

IQR = stat ['75%'] - stat ['25%']
upper = stat ['75%'] + 1.5 * IQR
lower = stat ['25%'] - 1.5 * IQR
print ('The lower and upper bound for suspected outliers in SalePrice are {} and {},'. format (upper, lower))


# In[ ]:


train[train.SalePrice == 3875.0]


# In[ ]:


train[train.SalePrice > 340075]


# lets not drop these observations since we may loose alot of information. Also, looking at the price corr to other feature graphs, the expensive house also rate high on other features hence may not be outliers as such.
# Note: This is an area we can review for better outcome

# In[ ]:


train.duplicated().sum()


# # 6. Feature Engineering

# In[ ]:


#get dummies for categorical data
train = pd.get_dummies(train)


# In[ ]:


# Display the first 5 rows of the last 12 columns to confirm that categorical features been converted to numerical 0 & 1
train.iloc[:,5:].head(5)


# In[ ]:


train.shape


# # 7. Train The Model

# In[ ]:


# define X and Y axis

X_train = train.drop(['SalePrice'], axis=1)
Y_train = train['SalePrice']


# In[ ]:


#Use numpy to convert to array
Xtrain = np.array(X_train)
Ytrain = np.array(Y_train)


# In[ ]:


# use decision tree

from sklearn.tree import DecisionTreeRegressor
decision_tree = DecisionTreeRegressor()
decision_tree.fit(Xtrain, Ytrain)
acc_decision_tree = round(decision_tree.score(Xtrain, Ytrain) * 100, 2)
acc_decision_tree


# In[ ]:


# use random forest
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(Xtrain, Ytrain);
rf


# In[ ]:


from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# # 8. Make Prediction On The Test Set

# ## Prepare the test set

# In[ ]:


# drop all features dropped during training

test = test.drop (['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage','GarageCond','GarageType','GarageYrBlt','GarageFinish','GarageQual','BsmtExposure','BsmtFinType2','BsmtFinType1','BsmtCond','BsmtQual','MasVnrArea','MasVnrType'], axis =1)


# In[ ]:


test.shape


# In[ ]:


#get dummies for categorical data
test = pd.get_dummies(test)


# In[ ]:


test.shape


# **To Continue...**...
