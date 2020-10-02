#!/usr/bin/env python
# coding: utf-8

# # Contents
# 
# * Intutive example
# * ID3 algorithm
# * Entropy
# * Information gain
# * Overfitting in Decision trees

# ## Intutive example
# 
# A decision tree is a graph that uses a branching method to illustrate every possible outcome of a decision.
# 
# Lets try to understand this using an example.
# 
# We have collected 2 weeks of data for a player with weather information. Using this example we will build Decision tree which will help us to identify outcome of new case.

# ![](d1.png)

# For out understanding lets just start with Outlook and see how data splits 
# 
# ![](d2.png)
# 
# Overcast has **pure ** subset, so we don't have to do anyting further to it. The other subsets are mixed so we will further split them. 

# Next for Sunny let pick humid for split and for Rain let put Wind for split
# ![](d3.png)

# Now based on above splits we have achieved subsets which are pure. So our decision tree will look like:
# ![](d4.png)
# 
# Here, tree dosen't need to be balanced. Some nodes can go deeper than other. 

# In the process we remember the count of outcome at each node. This is helpful when we **Prune** the tree.
# ![](d5.png)
# 
# Instead of deviding tree on Humidity we may decide to cut the tree off and we would end up in unpure set. So how we would identify if a case fall in left side of tree?
# 
# As player played in 2 days and not played in 3 days, so he is most likely not to play 

# ## ID3 algorithm
# 
# The algorithm which builds this data structure is called ID3 algorithm and it is a recursive algorithm.
# ![](d6.png)

# ** How do you pick the best attribute? **
# 
# You can randomly pick any atribute and it will result on different partition of data.
# 
# ![](d7.png)
# 
# In general you want split from decision tree which is heavily biased to either Positive or Negative outcome.
# 
# We want some measure/formula which gives information about purity of subsets

# ## Entropy
# 
# There are many different ways to define purity of subsets one way to define it is Entropy.
# 
# ![](d8.png)
# 
# example:
# 
# ![](d9.png)
# 
# Entropy can be understood with below graph, if set is purely negative or postive it will have lower number on the other hand if set is balanced it will hive higher number
# ![](d10.png)

# ## Information Gain
# 
# Entropy gave information about single subset, but when building decision tree we can have multiple subsets at each node.
# 
# We are going to take average at each splits.
# 
# ![](d11.png)
# 
# ![](d12.png)
# 
# ![](d13.png)
# 
# * Gains tells you how certain positive / negative become when you pick an attribute
# * So you have to pick every attribute and compute IG for each attribute and then select the attribute that has highest IG
# * You do this process recursively, you do it to root then to child and go on till you find best split
# * You do this process one level split at a time
# 
# 

# ## Comparison of Entropy, Gini Index and Classification Error
# ![](d15.png)
# 
# ### Classification Error
# ![](d16.png)
# 
# ### Gini Index
# ![](d17.png)
# 
# ### Entropy
# ![](d18.png)
# 
# ![](d19.png)

# ## Overfitting in decision trees
# 
# The good and bad thing about decision tree is that it will always partition perfectly. Because you will keep splitting the data until you get ** singletons **. The singletons instance will always be pure as they will either contain negative or positive instance.
# 
# ![](d14.png)

# In[ ]:




