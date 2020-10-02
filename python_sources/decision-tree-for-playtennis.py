#!/usr/bin/env python
# coding: utf-8

# # **Implementing Decision Tree using Scikit Learn**
# 
# This notebook is a reference notebook to a blog, [Decision Tree for Beginers](https://link.medium.com/wLnRkIdpR3). 

# In[ ]:


#numpy and pandas initialization
import numpy as np
import pandas as pd


# In[ ]:


#Loading the PlayTennis data
PlayTennis = pd.read_csv("../input/PlayTennis.csv")


# In[ ]:


PlayTennis


# It is easy to implement Decision Tree with numerical values. We can convert all the non numerical values into numerical values using [LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)

# In[ ]:


from sklearn.preprocessing import LabelEncoder
Le = LabelEncoder()

PlayTennis['outlook'] = Le.fit_transform(PlayTennis['outlook'])
PlayTennis['temp'] = Le.fit_transform(PlayTennis['temp'])
PlayTennis['humidity'] = Le.fit_transform(PlayTennis['humidity'])
PlayTennis['windy'] = Le.fit_transform(PlayTennis['windy'])
PlayTennis['play'] = Le.fit_transform(PlayTennis['play'])


# In[ ]:


PlayTennis


# * Lets split the training data and its coresponding prediction values.
# * y - holds all the decisions.
# * X - holds the training data.

# In[ ]:


y = PlayTennis['play']
X = PlayTennis.drop(['play'],axis=1)


# In[ ]:


# Fitting the model
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, y)


# In[ ]:


# We can visualize the tree using tree.plot_tree
tree.plot_tree(clf)


# [GraphViz](https://www.graphviz.org/) gives a better and clearer Graph.

# In[ ]:


import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph


# In the above graph,
# 
# * X[0] -> Outlook
# * X[1] -> Temperature
# * X[2] -> Humidity
# * X[3] -> Wind
# 
# values![image.png](attachment:image.png)
# 
# 

# Since we dont have any data to test. we can just make the model to predict our train data.

# In[ ]:


# The predictions are stored in X_pred
X_pred = clf.predict(X)


# In[ ]:


# verifying if the model has predicted it all right.
X_pred == y

