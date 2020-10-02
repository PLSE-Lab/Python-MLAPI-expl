#!/usr/bin/env python
# coding: utf-8

# ## In this notebook, we will see how Decision tree classifier performs on a data which is not linearly separable. 

# In[ ]:


# Essential libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier, plot_tree


# IPython libraries

from ipywidgets import interactive
from IPython.display import display
import ipywidgets as widgets


# ### Now lets first load the circles dataset into objects and visualize it 

# In[ ]:


data = make_circles(n_samples=100, shuffle=True, noise=0, random_state=2020)
X,Y = data


# In[ ]:


# Lets plot our circles

plt.figure(figsize=(8,5))
plt.scatter(X[:,0],X[:,1],c=Y,s=200,edgecolors='k')
plt.xlabel('x1',fontsize=14)
plt.ylabel('x2',fontsize=14)
plt.grid(True)
plt.show()


# In[ ]:


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
m = interactive(decision_tree,max_depth=widgets.IntSlider(min=1,max=8,step=1,description= 'Max Depth',
                                       stye=style,continuous_update=False))

# Set the height of the control.children[-1] so that the output does not jump and flicker
output = m.children[-1]
output.layout.height = '350px'

# Display the control
display(m)


# ### Lets now load the moon dataset into objects and visualize it 

# In[ ]:


data = make_moons(n_samples=100, shuffle=True, noise=0.15, random_state=2020)
X,Y = data


# Lets plot our moons

plt.figure(figsize=(8,5))
plt.scatter(X[:,0],X[:,1],c=Y,s=200,edgecolors='k')
plt.xlabel('x1',fontsize=14)
plt.ylabel('x2',fontsize=14)
plt.grid(True)
plt.show()



# In[ ]:


style = {'description_width': 'initial'}
m = interactive(decision_tree,max_depth=widgets.IntSlider(min=1,max=8,step=1,description= 'Max Depth',
                                       stye=style,continuous_update=False))

# Set the height of the control.children[-1] so that the output does not jump and flicker
output = m.children[-1]
output.layout.height = '350px'

# Display the control
display(m)


# ## In the both plots, we see that Decision Tree classifier is able to classify both circles and moons but not very accurately. This is because the Decision trees love orthogonal decision boundaries i.e. all splits are perpendicular to an axis.
