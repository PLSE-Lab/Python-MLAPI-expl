#!/usr/bin/env python
# coding: utf-8

# Loading all the dataset ;
# 1. Pandas for loading excel
# 2. sklearn for ML algos

# In[ ]:


import pandas as pd
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# **> Loading Dataset and Dropping all rows with missing fields **

# In[ ]:


train = pd.read_csv("/kaggle/input/titanic/train.csv")
test= pd.read_csv("/kaggle/input/titanic/test.csv")
results = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
test['Survived'] = results['Survived'].tolist()

train.dropna(inplace= True)
test.dropna(inplace= True)


# Taking all important features.

# In[ ]:


x_train = train[['Pclass','SibSp','Parch','Fare','Sex','Age','Embarked']]
x_test  = test[['Pclass','SibSp','Parch','Fare','Sex','Age','Embarked']]
y_train = train[['Survived']]
y_test = test[['Survived']]
print(x_train)
print(x_test)


# In[ ]:


x_train = pd.get_dummies(x_train)
x_test = pd.get_dummies(x_test)
print(x_train)
print(x_test)


# In[ ]:


tree_classifier = DecisionTreeClassifier(max_depth=50,max_leaf_nodes =4, random_state=7)
tree_classifier.fit(x_train, y_train)


# Testing the accuracy

# In[ ]:


tree_score_test = tree_classifier.score(x_test,y_test)
print(tree_score_test)


# In[ ]:




