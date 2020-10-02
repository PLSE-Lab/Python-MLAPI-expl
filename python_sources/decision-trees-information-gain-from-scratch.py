#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
from collections import Counter

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/mushrooms.csv')
data.head()


# In[ ]:


data.isnull().sum()


# The idea behind building trees is, finding the best feature to split on that generates the largest information gain or provides the least uncertainity in the following leafs.
# The 2 most known ways to find these features are:
# 1. Gini Impurity
# 2. Entropy
# 
# I will give a brief about the Gini Impurity method, but will implement the tree using calculated Entropy.

# **Gini Impurity:**
# 
# We start at 1 (maximum impurity) and subtract the squared percentage of each label in the data set. As an example, if a data set had 5 items of class (1) and 5 item of class (0), the Gini impurity of the set would be
# 
# \\(G = 1 - (5/10)^2 - (5/10)^2  \\)
# 
# That is the impurity at any given instance/leaf, to find the Weighted Information Gain, you start with the old (root) Gini Impurity and subtract the sum of all weighted Gini Impurities.
# The weighted Gini Impurity is the same as the Gini Impurity but multiplied by a ratio (weight) which is the number of data points in the new leaf divided by the number of points in the original root.
# ![](https://drive.google.com/uc?id=1Kf855C45vVGh-1-FOVaCTgYp6MdEyz3M)
# The Weighted Gain for this example \\( = 0.5 - (2/10)*0.5 - (5/10)*0.48 - (4/10)*0.44 = 0.026  \\)
# 
# 
# **Entropy:**
# Similar to the Gini Impurity but follows a different formula for the calculation:
# ![](https://drive.google.com/uc?id=1m0W6QKmGbS6GTSDo1zyogpOo9eDpqIZp)
# We will still find the information gain, using weighted entropies and pick the attribute which provided the maximum information gain.
# ![](https://drive.google.com/uc?id=14xKt2BkbU1qjFwLZ-k16SLf3IJtTrPWA)

# In[ ]:


print('We have {} features in our data'.format(len(data.columns)))


# Let's start by creating 2 functions, one that calculates the entropy and one that calculaten the information gain.
# 

# In[ ]:


def entropy(labels):
    entropy=0
    label_counts = Counter(labels)
    for label in label_counts:
        prob_of_label = label_counts[label] / len(labels)
        entropy -= prob_of_label * math.log2(prob_of_label)
    return entropy

def information_gain(starting_labels, split_labels):
    info_gain = entropy(starting_labels)
    for branched_subset in split_labels:
        info_gain -= len(branched_subset) * entropy(branched_subset) / len(starting_labels)
    return info_gain


# In[ ]:


def split(dataset, column):
    split_data = []
    col_vals = data[column].unique() # This tree generation method only works with discrete values
    for col_val in col_vals:
        split_data.append(dataset[dataset[column] == col_val])
    return(split_data)


# In[ ]:


def find_best_split(dataset):
    best_gain = 0
    best_feature = 0
    features = list(dataset.columns)
    features.remove('class')
    for feature in features:
        split_data = split(dataset, feature)
        split_labels = [dataframe['class'] for dataframe in split_data]
        gain = information_gain(dataset['class'], split_labels)
        if gain > best_gain:
            best_gain, best_feature = gain, feature
    print(best_feature, best_gain)
    return best_feature, best_gain

new_data = split(data, find_best_split(data)[0]) # contains a list of dataframes after splitting


# Now, a recursive call is needed to keep finding best split for every new data subset. I will stop here as it would get rather complex and the goal of this kernel was just to practice the idea behind decsion trees in general and calculating the entropy.
