#!/usr/bin/env python
# coding: utf-8

# My idea was to find out the **training samples with the same feature sets but with different target values**. That will help me to **remove the data ambiguity** and let my models perform better on the leaderboard. And I found some! 
# 

# In[ ]:


# Reading the data
import pandas as pd 
train = pd.read_csv("../input/train.csv", index_col='ID')

# Remove target to simplify the calculation of duplicates
target = train.pop("target")

# Find duplicate rows
t = train.duplicated(keep=False)
duplicated_indexes = t[t].index.values
print("Indexes of duplicated rows: {}".format(duplicated_indexes))

# Show target values for selected indexes
target.loc[duplicated_indexes]


# The key idea is that we can't predict target variable for sure for these users. Moreover, we can't train on that data, because these users features are the same. 
# 
# The possible ways to solve this problem are:
# * Ask competition organizers to provide Kagglers with more features (however, I'm sure they won't do it),
# * Drop these users from training dataset to remove the ambiguity (but our training dataset is already very small)
# * Leave only one user in the dataset and change the target variable for him.
# 
# There are two basic approaches to how we can choose the best target variable for the remaining user.
# 
# * Just mean value:
# $$\text{New target} = \frac{10000000.0 + 20000000.0}{2} = 15000000$$
# 
# * Logarythmic mean:
# $$\text{New target} = \exp{\frac{\log{10000000.0} + \log{20000000.0}}{2}} \approx 14142135.623730922$$
# 
# The implementation of the last approach:

# In[ ]:


import numpy as np
first_ind, second_ind = duplicated_indexes[0], duplicated_indexes[1]
new_target_val = np.exp((np.log(target.loc[first_ind]) + np.log(target.loc[second_ind])) / 2)

target = target.drop(first_ind)
target[second_ind] = new_target_val
train = train.drop(first_ind)


# One of the possible continuations of this research is to try to remove the columns containing almost all zeros and look at the target spread of the obtained duplicate rows, if any.
