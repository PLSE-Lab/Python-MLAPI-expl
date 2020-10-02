#!/usr/bin/env python
# coding: utf-8

# ### In this notebook, we will use Decision Tree classifier to classify the three types of iris flowers based on the length of sepal and petal. We will also see how tuning the maximum depth of tree affects our classification
# 
# #### Ref : https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html#sphx-glr-auto-examples-tree-plot-iris-dtc-py

# In[ ]:


# Essential libraries

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree


# IPython libraries

from ipywidgets import interactive
from IPython.display import display
import ipywidgets as widgets


# In[ ]:


# Lets load iris data

iris = load_iris()


# ### Iris dataset contains following attributes :
# 
# ###    1. sepal length in cm
# ###   2. sepal width in cm
# ###   3. petal length in cm
# ###   4. petal width in cm
# ###   5. class: 
# ###      -- Iris Setosa
# ###      -- Iris Versicolour
# ###      -- Iris Virginica

# ### We will try to classify based on sepal and petal length

# In[ ]:


X = iris.data[:,[0,2]] # sepal and petal length
Y = iris.target  # Labels


# In[ ]:


# Lets train 

def decision_tree(max_depth):

    classifier = DecisionTreeClassifier(max_depth = max_depth)
    classifier.fit(X,Y)
    # Plotting the decision boundary

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = classifier.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx,yy,Z,alpha=0.4)
    plt.scatter(X[:,0], X[:,1], c = Y, s = 20, edgecolor = 'k')
    plt.xlabel("Sepal length (cm)")
    plt.ylabel("Petal length (cm)")
    plt.show()


# In[ ]:


style = {'description_width': 'initial'}
m = interactive(decision_tree,max_depth=widgets.IntSlider(min=1,max=5,step=1,description= 'Max Depth',
                                       stye=style,continuous_update=False))

# Set the height of the control.children[-1] so that the output does not jump and flicker
output = m.children[-1]
output.layout.height = '350px'

# Display the control
display(m)


# ### Similarly, one can use different features to classify iris. 
# 
# ### One important point to note from the above plot is that Decision trees love orthogonal decision boundaries i.e. all splits are perpendicular to an axis.

# ### In the next notebook, we will see how Decision tree classifier is not a good choice when the data is not linearly separable...
