#!/usr/bin/env python
# coding: utf-8

# # Decision Trees

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#imorting dataset
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()


# In[ ]:


type(iris)


# In[ ]:


iris.keys()


# In[ ]:


iris.data[:5]


# In[ ]:


iris.feature_names


# In[ ]:


X = iris.data[:,2:] #petal_lenght and petal width
y = iris.target


# In[ ]:


tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X,y)


# In[ ]:


from sklearn.tree import export_graphviz

#generate a dot fil for visualizing

export_graphviz(
    tree_clf,
    out_file = 'iris_tree.dot',
    feature_names = iris.feature_names[2:],
    class_names = iris.target_names,
    rounded = True,
    filled = True
)


# In[ ]:


#converting .dot file to .ng
get_ipython().system('dot -Tpng iris_tree.dot -o iri_tree1.png -Gdpi=600')


# In[ ]:


get_ipython().system('dir')


# <img src='iri_tree1.png'>
# 
# #### Decision tress do not require data preparation like feature scaling or centering of the data

# In[ ]:


#function to decision boundary
def plot_decision_boundary(clf, X, y, axes=[0, 7, -0, 3], iris = True, legend = True,plot_training=True):
    X1s = np.linspace(axes[0], axes[1], 100)
    X2s = np.linspace(axes[2], axes[3], 100)
    X1,X2 = np.meshgrid(X1s,X2s)
    X_new = np.c_[X1.ravel(),X2.ravel()]
    y_pred = clf.predict(X_new).reshape(X1.shape)
    plt.contourf(X1,X2,y_pred,alpha=0.3)
    
    if not iris:
        plt.contourf(X1,X2,y_pred,alpha=0.8)
    if  iris:
        plt.xlabel('Petal_lenght',fontsize=14)
        plt.ylabel('Petal_width',fontsize=14)
    else:
        plt.xlabel(r'$x_1$',fontsize=14)
        plt.xlabel(r'$x_2$',fontsize=14)
    if legend:
        plt.legend(loc='lower right',fontsize=14)
    if plot_training:
        plt.plot(X[:,0][y==0],X[:,1][y==0], 'ro',label='Setosa')
        plt.plot(X[:,0][y==1],X[:,1][y==1], 'ys',label='Versicolor')
        plt.plot(X[:,0][y==2],X[:,1][y==2], 'g^',label='Virginica')
        plt.axes(axes)


# In[ ]:


plt.figure(figsize=(12,6))
plot_decision_boundary(tree_clf,X,y)
plt.show()


# ### Estimating Class's probability

# In[ ]:


#prob across all class
tree_clf.predict_proba([[4.2,1.5]])


# In[ ]:


#prob the class of new instance
tree_clf.predict([[4.2,1.5]])


# class 1 is Versicolor

# ## CART Algorithm
# #### Classificatin And Regression Tree(CART)
# 
# Sklearn uses CART algorithm to train Decision Tress.
# **[idea]** the alog first slpits the train set into two subsets usig a single feature $k$ and threadhold value $t_{k}$
# (for example, "Petal_length <0.8")
# 
# it searched for the pair ($k,t_{k}$) that produces purest subset(pure == homogenous class, where impurity =0)
# 
# Cost function for CART is 
# $$J(k,t){k}) = \frac{m_{left}}{m}G_{left} + \frac{m_{right}}{m}G_{right}$$
# 
# where, $G_{left/right}$ measures the impurity in left or right subset<br>
#                $m_{left/right}$ meaures the instances of left/ right hand datasets

# ## Regulariation of Hyperparameter
# - `non parametric model`
# - `parametric model`
# non parametric model are prone to over fittig
# 
# `To avoid over fitting you need to resctict the Decisions tres's degree of freedom during training` this is what we call Regularization
# 
# some of the common reularization hyperparameter the help if regularizing the model:<br>
# - `max_depth` (restrict the maximum depth of the tree)<br>
# - `min_sample_split`(min. no. of samples a node must have before it can split)<br>
# - `min_sample_leaf`(min. no.of samples a leaf must have before it can split)<br>
# - `min_weight_fraction_leaf`(same as min_samples_leaf, except its expressed as a fraction of total no. of weighted instances)<br>
# - `max_leaf_node` (Max no. of leaf nodes)<br>
# - `max_features`(max no. of features to evaluate for splitting for each node)

# ## Decision creteria:
# 
# for splitting any node further into child/leaf nodes:
# * Gini index(default)
# * information gain
# * Chi-sqaure
# * Entropy
# 
# Read the math behind each of these criteria [here](https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/)
# 
# hyperparameter in Decision Tree to tweak for using other criteria for spliting the nodes:
# `criterion` hyperparamter
# 
# [NOTE]Gini impurity is faster to compute, so its a good default. However, Gini impurity tends to isolate the most frequent class in its own branch of the tree, while Entropy tends to produce a slightly more balanced tree
# 
# 

# In[ ]:


from sklearn.datasets import make_moons

Xm, ym = make_moons(n_samples=100, noise=0.25, random_state=29)

deep_tree_clf1 = DecisionTreeClassifier(random_state=29) # non parametric model
deep_tree_clf2 = DecisionTreeClassifier(min_samples_leaf=4, random_state=29) # parametric model
deep_tree_clf1.fit(Xm, ym)
deep_tree_clf2.fit(Xm, ym)

plt.figure(figsize=(11, 4))
plot_decision_boundary(deep_tree_clf1, Xm, ym, axes=[-1.5, 2.5, -1, 1.5], iris=False)
plt.title("No restrictions", fontsize=14)
plt.show()


# In[ ]:


plt.figure(figsize=(11, 4))
plot_decision_boundary(deep_tree_clf2, Xm, ym, axes=[-1.5, 2.5, -1, 1.5], iris=False)
plt.title("min_samples_leaf = {}".format(deep_tree_clf2.min_samples_leaf), fontsize=14)
plt.show()


# ## Regression Trees
# 
# the tree looks very similar to the classification tree. The main difference is that instead of predicting a class in each node, it predicts a value and the 
# `prediction is simply an average target value of all the training instances associated to that particular node.`
# 
# cost function to minimize for CART based algo for regression like tasks:
# 
# $$J(k,t_{k}) = \frac{m_{left}}{m}MSE_{left} + \frac{m_{right}}{m}MSE_{right}$$
# 

# In[ ]:


# quadratic training set + noise
np.random.seed(29)

m = 200
X = np.random.rand(m, 1)
y = 4 * (X - 0.5) ** 2
y = y + np.random.randn(m, 1)/10


# In[ ]:


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(max_depth=2,random_state=12)
tree_reg.fit(X,y)


# In[ ]:


tree_reg1 = DecisionTreeRegressor(random_state=29, max_depth=2)


tree_reg1.fit(X, y)


# In[ ]:


def plot_regressor(clf,X,y,axes = [0, 1, -0.2, 2]):
    X1 = np.linspace(axes[0],axes[1],500).reshape(-1,1)
    y_pred = tree_reg.predict(X1)
    plt.axis(axes)
    plt.xlabel("$x_1$", fontsize=14)
    plt.plot(X, y, "b.")
    plt.plot(X1, y_pred, "r.-", linewidth=2, label="$\hat{y}$")


# In[ ]:


plt.figure(figsize=(11,5))
plot_regressor(tree_reg1,X,y)
plt.show()


# In[ ]:


tree_reg2 = DecisionTreeRegressor(random_state=2, max_depth=6)
tree_reg2.fit(X, y)


# In[ ]:





# In[ ]:




def plot_regression_predictions(tree_reg, X, y, axes=[0, 1, -0.2, 1], ylabel="$y$"):
    x1 = np.linspace(axes[0], axes[1], 500).reshape(-1, 1)
    y_pred = tree_reg.predict(x1)
    plt.axis(axes)
    plt.xlabel("$x_1$", fontsize=14)
    if ylabel:
        plt.ylabel(ylabel, fontsize=14, rotation=0)
    plt.plot(X, y, "b.")
    plt.plot(x1, y_pred, "r.-", linewidth=2, label="$\hat{y}$")

plt.figure(figsize=(11, 5))
plt.subplot(121)
plot_regression_predictions(tree_reg1, X, y)
for split, style in ((0.1973, "k-"), (0.09, "k--"), (0.7, "k--")):
    plt.plot([split, split], [-2, 2], style, linewidth=2)
plt.text(0.21, 0.65, "Depth=0", fontsize=14)
plt.text(0.01, 0.2, "Depth=1", fontsize=14)
plt.text(0.65, 0.8, "Depth=2", fontsize=14)
plt.title("max_depth=2", fontsize=14)

plt.subplot(122)
plot_regression_predictions(tree_reg2, X, y)
for split, style in ((0.1973, "k-"), (0.09, "k--"), (0.7, "k--")):
    plt.plot([split, split], [-2, 2], style, linewidth=2)
plt.text(0.3, 0.5, "Depth=2", fontsize=14)
plt.title("max_depth=3", fontsize=14)

plt.show()


# In[ ]:




