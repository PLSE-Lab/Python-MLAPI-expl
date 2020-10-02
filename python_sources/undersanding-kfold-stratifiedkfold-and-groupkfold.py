#!/usr/bin/env python
# coding: utf-8

# The most common ways to overfit data is to train and test in the same data. One solution to solve this problem is to use the cross validation.
# 
# Commonly used cross validation strategies are `KFold`, `Stratified KFold` and `GroupKFold`.

# ## KFold
# KFold divides all the samples in k groups of samples, called folds ( if k=n this is equivalent to the Leave One Out strategy), of equal sizes (if possible). The prediction function is learned using k - 1 folds, and the fold left out is used for test.

# In[ ]:


from sklearn.model_selection import KFold
X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
kf = KFold(n_splits=3, shuffle=True)
for train_index, test_index in kf.split(X):
     print("%s %s" % (train_index, test_index))


# ## Stratified KFold
# If we use the KFold for unbalanced dataset, we might endup in a training data that conatins no or very few minority classes. To avoid this problem, we use the stratified KFold. StratifiedKFold is a variation of k-fold which returns stratified folds: each set contains approximately the same percentage of samples of each target class as the complete set.

# In[ ]:


from sklearn.model_selection import StratifiedKFold
X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
skf = StratifiedKFold(n_splits=4, shuffle=True)
for train_index, test_index in skf.split(X, y):
    print("%s %s" % (train_index, test_index))


# Here if you carefully observe you can see that the test split has atleast one y which has value 1

# ## GroupKFold
# There could me one more situation where we are collecting data from individuals that belong to multile groups. Suppose we want to train on some individuals belonging to a set groups and test on remaining set of groups. In this situation, we can use GroupKfold. The GroupKFold need groups also as an input.

# In[ ]:


from sklearn.model_selection import GroupKFold
X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
groups = ['a','a', 'a', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'd', 'd', 'd', 'd', 'd', 'd']
gkf = GroupKFold(n_splits=4)
for train_index, test_index in gkf.split(X, y, groups=groups):
     print("%s %s" % (train_index, test_index))


# Here if you carefully observe, you can see that the test_index belongs to one group. For the first case its the group `d`, second case its group `c`, third case its the group `b` and in the fourth case its the group `a`.
