#!/usr/bin/env python
# coding: utf-8

# # How to plot a Decision Tree and a path from a particular prediction ?
# 
# I was looking online how to plot a particular prediction and a DecisionTreeClassifier from scikit-learn without [graphviz](https://graphviz.org/) or with dot command line (C.F. [Introduction: How to Visualize a Decision Tree in Python using Scikit-Learn](https://www.kaggle.com/willkoehrsen/visualize-a-decision-tree-w-python-scikit-learn)).
# 
# At the end I found the [`plot_tree`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html) function from pandas (scikit-learn >= 0.21.3).
# 
# **Don't forget that with the goal to get an interpretable model it can be usefull !**
# 
# # Load data : Iris
# For this notebook, I prefer to use an easy dataset (that you may know well) : iris dataset

# In[ ]:


from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()

# I prefer to work with DataFrame object so I convert the data into a DataFrame
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target


# In[ ]:


iris_df.head()


# # Model : Decision Tree
# Let's create a simple `DecisionTreeClassifier` and fit it with our data.

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
model.fit(iris.data, iris.target)


# # Use plot_tree function
# Now let's dive into the main part of this notebook : I'm creating a function called `plot_decision_tree` that create a visualization of a Decision tree and I choose to return it because after plotting the full tree, **[spoiler alert]** we will use it to remove nodes and leaves that are useless.

# In[ ]:


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

def plot_decision_tree(model, feature_names, class_names):
    # plot_tree function contains a list of all nodes and leaves of the Decision tree
    tree = plot_tree(model, feature_names = feature_names, class_names = class_names,
                     rounded = True, proportion = True, precision = 2, filled = True, fontsize=10)
    
    # I return the tree for the next part
    return tree


# In[ ]:


fig = plt.figure(figsize=(15, 12))
plot_decision_tree(model, iris.feature_names, iris.target_names)
plt.show()


# # Plotting a particular prediction path
# To meet this goal I had to think differently about `matplotlib`, for many use of this package you have to create the graphic from scratch.
# 
# Here it's different : you have the graph but you it to edit it (and remove what you do not need). So let's take a look of the code below.

# In[ ]:


def plot_decision_path_tree(model, X, class_names=None):
    fig = plt.figure(figsize=(10, 10))
    class_names = model.classes_.astype(str) if type(class_names) == type(None) else class_names
    feature_names = X.index if type(X) == type(pd.Series()) else X.columns
    
    # Getting the tree from the function programmed above
    tree = plot_decision_tree(model, feature_names, class_names)
    
    # Get the decision path of the wanted prediction 
    decision_path = model.decision_path([X])

    # Now remember the tree object contains all nodes and leaves so the logic here
    # is to loop into the tree and change visible attribute for components that 
    # are not in the decision path
    for i in range(0,len(tree)):
        if i not in decision_path.indices:
            plt.setp(tree[i],visible=False)

    plt.show()


# Let's use the (wonderful) function created above.

# In[ ]:


from IPython.display import display

display(iris_df.iloc[0,:].to_frame().T)
plot_decision_path_tree(model, iris_df.iloc[3,:-1], class_names=iris.target_names)


# In[ ]:


display(iris_df.iloc[50,:].to_frame().T)
plot_decision_path_tree(model, iris_df.iloc[50,:-1], class_names=iris.target_names)


# In[ ]:


display(iris_df.iloc[100,:].to_frame().T)
plot_decision_path_tree(model, iris_df.iloc[100,:-1], class_names=iris.target_names)


# Thank you for reading, I hope this will be usefull.
# 
# That's all folks !
# 
# $Nathan$
