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
import os
# Import 'tree' from scikit-learn library
from sklearn import tree

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
print(check_output(["ls"]).decode("utf8"))
train = pd.read_csv(os.path.join('../input', 'train.csv'))
test = pd.read_csv(os.path.join('../input', 'test.csv'))

print (train.info())
print (train.head())
# Any results you write to the current directory are saved as output.


# In[ ]:


sns.countplot(train['Sex'])


# In[ ]:


# Import 'tree' from scikit-learn library
from sklearn import tree
# Convert the male and female groups to integer form
train_0 = train.copy()
train_0["Sex"][train_0["Sex"] == "male"] = 0
train_0["Sex"][train_0["Sex"] == "female"] = 1
train_0["Age"].fillna(train_0["Age"].median())

print (train_0.info())
# Create the target and features numpy arrays: target, features_one
target = train_0['Survived'].values
features_one = train_0[["Pclass", "Sex", "Fare"]].values


# In[ ]:


# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# Look at the importance and score of the included features
print(my_tree_one.feature_importances_)
print(my_tree_one.score(features_one, target))


# In[ ]:


test_0 = test.copy()
test_0.Fare[152] = test_0['Fare'].median()
test_0["Sex"][test_0["Sex"] == "male"] = 0
test_0["Sex"][test_0["Sex"] == "female"] = 1
test_0["Age"].fillna(test_0["Age"].median())

print(test_0.info())
test_features = test_0[['Pclass', 'Sex', 'Fare']].values

#print(test_features.info())
my_prediction = my_tree_one.predict(test_features)
print (my_prediction)


# In[ ]:


PassengerId =np.array(test["PassengerId"]).astype(int)
print (PassengerId)
print (my_prediction)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
#my_solution['Survived'] = my_prediction
print(my_solution)

# Check that your data frame has 418 entries
print(my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])

