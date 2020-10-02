#!/usr/bin/env python
# coding: utf-8

# Using a Tree to Classify Whether Mushrooms Poisonous
# Written by:
# Ethan Bartlett

# 1. cleans data
# 2. splits data into test and train
# 3. creates a tree structure
# 4. trains tree on training data

# The Below code imports neccecary libraries
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import numpy as np # linear algebra
import pandas as pd # data orginization tool used for data processing/cleaning, CSV file I/O (e.g. pd.read_csv)
#Importining necceary sklearn libraries
from sklearn.preprocessing import LabelEncoder 
from sklearn.tree import DecisionTreeClassifier as skTree
from sklearn.model_selection import train_test_split 
from sklearn.tree import export_graphviz
#These Librares make it significantly easier to create graphs and other visual representations
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import graphviz

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#This code just downloads the dataset from kaggle and stores it as a pandas array 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
mushroom = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')
print(mushroom)
# Any results you write to the current directory are saved as output.


# The below code turns the strings in our data to integers

# In[ ]:


labelencoder = LabelEncoder()
for col in mushroom.columns:
    mushroom[col] = labelencoder.fit_transform(mushroom[col])
print(mushroom)


# Rearanges data into a form that sklearn can use

# In[ ]:


X = (mushroom.drop('class', axis = 1)).values
y = (mushroom['class']).values
print(X)
print("----")
print(y)


# prints tree

# In[ ]:


dot_data = export_graphviz(tree, out_file=None, feature_names = (mushroom.drop('class', axis = 1)).columns)
graph = graphviz.Source(dot_data) 
graph


# Below, you can see that all mushrooms with a gill-color of "0" are poisonous (class "1") (the first split), so entropy is 1. This is the biggest number (so it has the biggest impact).

# In[ ]:


pd.crosstab(mushroom['class'],mushroom['gill-color'])


# creates crosstab for model on test data
# 
# Rows are predictions in the test data and columns are truth. 1 is poisonous and 0 is edible.
# 
# The model predicts 40 edible mushrooms to be poisonous. And 12 poisonous mushrooms to be edible. By examining the tree above, you can come up with rules that are more accurate.

# In[ ]:


predict = model.predict(X_test)

pd.crosstab(predict,y_test)


# In[ ]:


#percent BAD WRONG
print(str(12/(1251+40+12+1135)*100)+"% Poisonous Incorrectly Predicted as Edible")


# creates crosstab for model on training data

# In[ ]:


predict = model.predict(X_train)

pd.crosstab(predict,y_train)

