#!/usr/bin/env python
# coding: utf-8

# ## Custom Imputer Class
# You'll hardly encounter a clean, complete dataset before training models. Most likely, you will have to fill in missing values for multiple columns where each feature might have a different imputation strategy. In addition, when a model is in production, the data coming in to make new predictions on might contain missing values as well so you will want to fill those in with the same imputation strategies that were used when cleaning the dataset before training.
# 
# This custom imputation class handles missing values with various imputation strategies all in one go. Details about how to initialize the class are below. Lastly, this class implements the common fit-transform functions to be able to integrate the imputation in a scikit-learn pipeline thereby transforming both training data and new data points coming in during production. 

# In[ ]:


import pandas as pd
from customimputerclass import CustomImputer


# Using the data from the Kaggle House Price Regression competition, let's look at the missing data in both the train and test sets. 
# 
# We can see that there are many missing values in both the train and test set that we will have to deal with in some manner. Another thing to point out is that the test set contains features with missing values that the train set does not e.g. MSZoning. Therefore, we want to think ahead and plan for any possible missing values in future data points/sets and how we want to deal with them for each particular feature. 

# In[ ]:


train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


train_missing = train_df.isnull().sum()
train_miss_perct = train_missing/len(train_df)
train_miss_perct[train_miss_perct > 0]


# In[ ]:


test_missing = test_df.isnull().sum()
test_miss_perct = test_missing/len(test_df)
test_miss_perct[test_miss_perct > 0]


# To simplify this presentation, let's narrow the datasets to a subset of the features that have missing values. We are not missing any Neighborhood values but I am including in the filtered dataset to present one of the capabilities of the imputer.

# In[ ]:


df1 = train_df[['LotFrontage', 'Alley', 'Fence', 'MSZoning', 'MSSubClass', 'Neighborhood']]
df2 = test_df[['LotFrontage', 'Alley', 'Fence', 'MSZoning', 'MSSubClass', 'Neighborhood']]


# To initialize a custom imputer, we need to pass in a dictionary of the columns and their respective fill strategies. The fill strategies mimic what's available in the scikit-learn SimpleImputer class.
# 
# The fill strategy options are:
# <br>1: mean
# <br>2: median
# <br>3: most frequent
# <br>4: group by a certain column then using one of the strategies in 1-3
# <br>5: any manual value we want to impute with
# 
# Below we create this dictionary. The keys are the columns to impute and the values are the imputation strategy. I won't show an example for median because the implementation is the same as mean. 
# 
# A couple of notes for MSZoning...
# <br>1: We want to fill these missing values based on the most frequent value in the MSSubClass the data point belongs to. 
# <br>2: Again, this feature is complete in the train set but not the test set so we are still including in the dictionary; there will be no effect on the train set since the values are all present but we still want to prepare for the chance that the column will have nulls. 

# In[ ]:


fill_vals = {'LotFrontage': 'mean', 'Alley': 'most_frequent', 'Fence': 'None', 'MSZoning': ('MSSubClass', 'most_frequent')}


# First focusing on LotFrontage and Alley, the mean of LotFrontage is 70.05 (of course, when not including nulls) and the most frequent value of Alley is Grvl. According to the description of the dataset, if Fence is null, the property does not have a fence so we will replace nulls with 'None'. Below we can see some rows of the dataset where LotFrontage, Alley, and Fence are missing. 

# In[ ]:


print(df1['LotFrontage'].mean())
print(df1['Alley'].mode())
df1[df1['LotFrontage'].isnull()]


# In[ ]:


imputer = CustomImputer(fill_vals)
X = imputer.fit_transform(df1)


# Taking a look at a subset of some of the rows with missing values above, we can see that LotFrontage was filled in with the mean, Alley was filled in with the most frequent value, and Fence was filled in with None!

# In[ ]:


X.iloc[[7, 12, 1446], :]


# Now let's take a look at the missing MSZoning in the test set. We see that we have rows with missing MSZoning where MSSubClass is 20, 30, and 70. The most frequent MSZoning values of each of these MSSubClass values is RL, RM, and RM, respectively.

# In[ ]:


df2[df2['MSZoning'].isnull()]


# In[ ]:


df2.groupby(['MSSubClass', 'MSZoning'])['Neighborhood'].count()


# In[ ]:


X2 = imputer.transform(df2)


# We can see that along with the missing LotFrontage and Alley values, the imputer filled in the missing MSZoning values with the most frequent value with respect to the MSSubClass category. 

# In[ ]:


X2.iloc[[455, 756, 790, 1444], :]


# I hope this helps quicken the process of imputing missing data. For the features we use in our model, we can ask ourselves how we'd want to impute each feature in the case there are missing values present, in the training set, test set, and new data points coming in later in the future. Even if they're not missing in the train/test set doesn't mean they won't be missing later on. 

# In[ ]:




