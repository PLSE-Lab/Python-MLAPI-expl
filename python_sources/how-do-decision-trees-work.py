#!/usr/bin/env python
# coding: utf-8

# # How Do Decision Trees Work?
# 
# Decision Trees (DT's) are one of my favorite Machine Learning models. They're simple to understand, versatile, easily visualized and above all simple to use. DT's are also really easy to explain to other people. By using a DT you can show someone that Machine Learning doesn't have to be a black box that gets input and spits out some seemingly random output. 
# 
# ## What is a Decision Tree?
# To understand what a Decision Tree is, it is easiest to just show it in action. We'll import the iris dataset and fit a Decision Tree on the data. Then we'll visualize it with the graphviz package.

# In[ ]:


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target

clf = DecisionTreeClassifier(max_depth=2)
clf.fit(X,y)


# In[ ]:


from sklearn.tree import export_graphviz
import graphviz

dotdata = export_graphviz(
    clf,
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True,
)
graphviz.Source(dotdata) 


# The Decision Tree we have generated is readable by humans. This means that we can get an idea of how the model works by reading it.
# 
# ## How do you read a Decision Tree?
# To read a Decision Tree you start at the top. This is the first (0th) layer. Every box is a _node_, these _nodes_ contain a conditional statement (like an if - statement), a gini value (more on that later), a sample size value which tells us how many instances of the data have passed this node, value tells you how many of these samples fall into the different classes and class tells you which class the DT thinks instances are that pass this point.
# 
# At every node you evaluate the condition of the node and choose a path down based on this decision. You never go up the tree, only down it. When you reach the bottom of the tree you have reached your conclusion about which class the instance you are looking at is.
# 
# ## Gini
# Gini is a value used to measure the _purity_ of a node. A _pure_ node (gini = 0) is a node that only has instances belonging to the same class.
# 
# You can calculate gini using the following formula:
# 
# $ G_i = 1 - \sum_{k=1}^{n} {p_{i,k}}^2$
# 
# Where:
# * $G_i$ = Gini Impurity
# * $p_{i,k}$ = The ratio of class $k$ instances among training instances of the $i^{th}$ node
# 
# ## How does it work?
# The Decision Trees from the Scikit-Learn library are built using the CART (Classification and Regression Tree) algorithm. It performs a few steps to generate a node and repeats this process as needed.
# 
# 1. Find the most significant feature $k$
# 1. Find the optimal treshold $t_k$
# 1. Split the data into 2 subsets based on $k$ and $t_k$
# 
# 
# ## Conclusion
# 
# So we now know some more about Decision Trees and how they work. This is important for the next step, ensemble learning. We'll discuss Random Forests and how they work by creating a lot of Decision Trees and subsets of data.
# 
# ### Previous Kernel
# 
# [What are Support Vector Machines?](https://www.kaggle.com/veleon/what-are-support-vector-machines)
# ### Next Kernel
# 
# [What is Ensemble Learning?](https://www.kaggle.com/veleon/what-is-ensemble-learning)
# 

# In[ ]:




