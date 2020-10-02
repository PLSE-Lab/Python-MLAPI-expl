#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Decistion three is the basis for all three-based methods, most notably XGBoost, etc. Have a solid understanding of how it works is crucial.
# 
# Here, we will consider the Iris data and try to interpret the decision tree and discuss how can we improve the prediction.
# 
# Firstly, some background about the data:

# In[10]:


from sklearn.datasets import load_iris
iris = load_iris()
print(iris['DESCR'])


# OK, let's do the model.

# In[11]:


X = iris.data
y = iris.target

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=2)
clf.fit(X, y)


# In[16]:


from sklearn import tree
import graphviz
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph


# **References**
# 
# http://www.saedsayad.com/decision_tree.htm

# In[ ]:




