#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## A glance at the target variable
# 
# First step is to really understand a bit more about the target variable:
# 
# 1. How is the variable distributed
# 2. To which variables it  correlates the most.
# 3. What's the target variable distribution given other indipedent variables.

# In[ ]:


#Loading Train and Test Data
train = pd.read_csv("../input/train.csv", parse_dates=["first_active_month"])
test = pd.read_csv("../input/test.csv", parse_dates=["first_active_month"])

print(f"Train set contains {train.shape[0]} observations and {train.shape[1]} variables")
print(f"Test set contains {test.shape[0]} observations and {test.shape[1]} variables")
train.head()


# In[ ]:


dupcheck = len(train) - len(train['card_id'].unique())
print(f"The data contains {dupcheck} duplicate card ids")


# The target variable seems to be pretty much normally distributed with some outliers on both sides. We will look at those at a later stage. The boxplot will help use realize that the distribution has the median at 0 and is  pretty much symmetric. 
# 
# It's possible tha target variable - the loyalty score - has been standardized by subtracting the mean and dividing by the standard deviation. It's hard to say by just looking at the summary statistics, since we only see the train set.

# In[ ]:


plt.figure(figsize = (16, 6))

plt.subplot(121)
plt.hist(train["target"], bins = 25)
plt.title("Target Variable Histogram")

plt.subplot(122)
sns.boxplot(data = train, y = "target")
plt.title("Target Variable Boxplot")

plt.show()

train["target"].describe()


# If we look at the first active month, the distribution of cards activated in any given period is identical across train and test set. Which means that the two sets have been properly stratified. There's a peak towards the end of 2017 and a sudden dip in the first two months of 2018.

# In[ ]:


train['first_active_month'].value_counts(normalize = True).sort_index().plot(figsize = (16,4), label = 'Train')
test['first_active_month'].value_counts(normalize = True).sort_index().plot(figsize = (16,4), label = 'Test')
plt.legend()
plt.title('Frequency of cards by first active month - Train vs Test Set')
plt.show()


# If we look at the loyalty score by the first month the card was active we may notice that this score tends to go up with time, with an increasing trend especially in the last months.
# 
# > This means that the cards activated towards mid-end 2017 tend to have  a **higher target value**
# 

# In[ ]:


train[['first_active_month', 'target']].groupby('first_active_month').agg(np.mean).plot(figsize = (16,4))
plt.title("Average Target Value by Card First Active Month")


train[['first_active_month', 'target']].groupby('first_active_month'
                                               ).agg(np.mean).rolling(5).mean().plot(figsize = (16, 4))
plt.title("5 Periods Average Target Value by Card First Active Month")


# If we look at the other three features across both the train and test set we can see that the two sets are well stratified. **Feature_1** has 5 level with a majority at level 3. **Feature_2** has 3 levels, while **Feature_3** is a binomial one. Haven't read all the discussion, but hard to say what those features are. 

# In[ ]:


feats = np.arange(1,4)

nrows = len(feats)
ncols = 2

plt.figure(figsize = (15,4.5*len(feats)))
idxs = np.arange(1,7).reshape(nrows, ncols)

for idx, i in enumerate(feats):
    
    f_name = f"feature_{i}"
    
    plt.subplot(len(feats), ncols, idxs[idx][0])
    train[f_name].value_counts(normalize=True).plot(kind = 'bar', color = 'blue', alpha = 0.5)
    plt.title(f"Train Set {f_name}")
    
    plt.subplot(len(feats), ncols, idxs[idx][1])
    train[f_name].value_counts(normalize=True).plot(kind = 'bar', color = 'green', alpha = 0.5)
    plt.title(f"Test Set {f_name}")

plt.suptitle('Features Distribution Across Train and Test Set')


# The boxplots would revel that the target variable is well distributed across the level of each of the three features.

# In[ ]:


plt.figure(figsize = (16,4))
sns.boxplot(data = train, x='feature_1', y ='target')


# In[ ]:


plt.figure(figsize = (16,4))
sns.boxplot(data = train, x='feature_2', y ='target')


# In[ ]:


plt.figure(figsize = (16,4))
sns.boxplot(data = train, x='feature_3', y ='target')


# If we look at how the target variable correlats to threee other plots it can be seen that the highest degree is with feature_1. All three features are inversely correlated the loyalty score.

# In[ ]:


train.corr()['target'].head(3).plot(kind='bar', color='blue', alpha = 0.5, figsize = (10,4))


# In[ ]:




