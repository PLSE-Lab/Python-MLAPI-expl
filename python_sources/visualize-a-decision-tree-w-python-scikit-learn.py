#!/usr/bin/env python
# coding: utf-8

# # Introduction: How to Visualize a Decision Tree in Python using Scikit-Learn
# 
# The title is pretty self explantory! 

# ### Data: Good Old Iris Dataset
# 
# The data does not matter for this example, feel free to use your own.

# In[ ]:


from sklearn.datasets import load_iris

iris = load_iris()


# # Model: Random Forest Classifier
# 
# We'll create two version, one where the maximum depth is limited to 3 and another where the max depth is unlimited. (You could use a single decision tree for this as well, it's just that I often use a random forest for modeling.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

# Limit max depth
model = RandomForestClassifier(max_depth = 3, n_estimators=10)

# Train
model.fit(iris.data, iris.target)
# Extract single tree
estimator_limited = model.estimators_[5]
estimator_limited


# In[ ]:


# No max depth
model = RandomForestClassifier(max_depth = None, n_estimators=10)
model.fit(iris.data, iris.target)
estimator_nonlimited = model.estimators_[5]


# ## Export Tree as .dot File
# 
# Use the `export_graphviz` functionality in scikit-learn. Format the decision tree however you like: I suggest trying a few different options.

# In[ ]:


from sklearn.tree import export_graphviz
export_graphviz(estimator_limited, out_file='tree_limited.dot', feature_names = iris.feature_names,
                class_names = iris.target_names,
                rounded = True, proportion = False, precision = 2, filled = True)


# In[ ]:


export_graphviz(estimator_nonlimited, out_file='tree_nonlimited.dot', feature_names = iris.feature_names,
                class_names = iris.target_names,
                rounded = True, proportion = False, precision = 2, filled = True)


# ## Convert to png from the command line
# 
# Use the `dot` utility (may need to install on your computer). You can change the options, but the only one I have altered is the dots per inch (resolution)

# In[ ]:


get_ipython().system('dot -Tpng tree_limited.dot -o tree_limited.png -Gdpi=600')


# In[ ]:


from IPython.display import Image
Image(filename = 'tree_limited.png')


# The no max depth version of the tree can be rather unwiedly if you are using a large tree (usually occurs with a lot of features).

# In[ ]:


get_ipython().system('dot -Tpng tree_nonlimited.dot -o tree_nonlimited.png -Gdpi=600')


# In[ ]:


Image(filename = 'tree_nonlimited.png')


# # Full Script (using call instead of ! for system commands)
# 
# This can be directly copied and pasted into Python and run (it looks better in a Jupyter Notebook)

# In[ ]:


from sklearn.datasets import load_iris
iris = load_iris()

# Model (can also use single decision tree)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10)

# Train
model.fit(iris.data, iris.target)
# Extract single tree
estimator = model.estimators_[5]

from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = iris.feature_names,
                class_names = iris.target_names,
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Convert to png
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in python
import matplotlib.pyplot as plt
plt.figure(figsize = (14, 18))
plt.imshow(plt.imread('tree.png'))
plt.axis('off');
plt.show();


# # Conclusions
# 
# Visualizing a single decision tree can help give us an idea of how an entire random forest makes predictions: it's not random, but rather an ordered logical sequence of steps. I would go so far as to say this is how a human reasons: a flowchart of questions and answers. Feel free to use and adapt this code as required.
# 
# Best,
# 
# Will

# In[ ]:




