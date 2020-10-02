#!/usr/bin/env python
# coding: utf-8

# # Using Scikit-learn to Implement a Simple Decision Tree Classifier

# Scikit-learn is library for the development of machine learning models, whether for regression or classification problems, that is as widely used as it is appreciated.  Today we will be implementing a simple decision tree model to classify iris plants into three species using the lengths and widths of their petals and sepals, a [classic data set](https://en.wikipedia.org/wiki/Iris_flower_data_set) in the machine learning community.
# 
# The iris samples in question fall into three species: *Iris setosa*, *Iris versicolor*, and *Iris virginica*.  At first glance, the three species can look remarkably like each other, and this is especially the case for the latter two.  However, thanks to the careful measurements taken and presented by Anderson and Fisher, an individual sample may be classified with a high degree of accuracy via machine learning models if the relevant measurements are taken.

# Decision tree models are the simplest form of tree-based models, and are arguably the simplest form of supervised multivariate classification models. A series of logical tests (generally in the form of boolean comparisons) are applied to the sample entries and their resulting subsets in turn to arrive at a final decision. It is very easy to visualize the decision process in a simple flowchart to trace the rational of every assignment made by a decision tree model, making it among the most interpretable of models.
# 
# The following code, strongly derived from [scikit-learn's documentation](https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html), shows how a simple decision tree model may be applied and visualized.

# We begin by running the standard import cell common to kaggle notebooks.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# To this we will add our needed skikit-learn libraries, as well as some plotting functionality from Matplotlib.

# In[ ]:


# Import the needed matplotlib functionality for scatter plot visualization.
import matplotlib.pyplot as plt
# import the needed dataset.
from sklearn.datasets import load_iris
# Import the model and an additional visualization tool.
from sklearn.tree import DecisionTreeClassifier, plot_tree


# Once imports are complete, we may define a few parameters for later use...

# In[ ]:


# Define a variable to establish three classes/species.
class_count = 3
# Define standard RGB color scheme for visualizing ternary classification in order to match the color map used later.
plot_colors = 'brg'
# Define marker options for plotting class assignments of training data.
markers = 'ovs'
# We also need to establish a resolution for plotting.  I favor clean powers of ten, but this is not by any means a hard and fast rule.
plot_res = 0.01


# ... and load the data set in question.

# In[ ]:


# Load the iris dataset from scikit-learn (note the use of from [library] import [function] above)
iris = load_iris()


# Let us now go about applying the decision tree model in a pairwise fashion to the available features to see some simple applications of how decision trees can assign classifications to individual samples.  Since the data set is small and we aren't concerned today with validation, we won't bother with a train-test split.

# In[ ]:


# Set the size of the figure used to contain the subplots to be generated.
plt.figure(figsize=(20,10))

# Create an empty list of models to store the results of each pairwise model fit.
models = []

# Use enumerate() to define the possible pairs of features available and iterate over each pair.
for pair_index, pair in enumerate([[0, 1], [0, 2], [0, 3], 
                                           [1, 2], [1, 3], 
                                                   [2, 3] ]):

    # We only take the two features corresponding to the pair in question...
    X, y = iris.data[:, pair] , iris.target
    
    # ... to fit the decision tree classifier model.
    model = DecisionTreeClassifier().fit(X, y)
    
    # Append the results to the models list
    models.append(model)
    
    # Establish a two row by three column subplot array for plotting.
    plt.subplot(2, 3, pair_index + 1)
    
    # Define appropriate x and y ranges for each plot...
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    # ... and use each range to define a meshgrid to use as the plotting area.
    xx, yy = np.meshgrid(np.arange(x_min, 
                                   x_max, 
                                   plot_res),
                         np.arange(y_min, 
                                   y_max, 
                                   plot_res) )
    # Use plt.tight_layout() to establish spacing of the subplots.
    plt.tight_layout(h_pad = 0.5, 
                     w_pad = 0.5, 
                       pad = 4.0 )
    
    # Predict the classification of each point in the meshgrid based on the calculated model above.
    # The numpy methods .c_() and .ravel() reshape our meshgrid values into a format compatible with our model.predict() method,
    Z = model.predict(np.c_[xx.ravel(), 
                            yy.ravel() ])
    # Reshape the predictions to match xx...
    Z = Z.reshape(xx.shape)
    # ... and prepare a contour plot that reflects the predictions .
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.brg)
    
    # Define the subplot axis labels after title casing while preserving case on the unit of measure 
    plt.xlabel(iris.feature_names[pair[0]].title()[0:-4] + iris.feature_names[pair[0]][-4:])
    plt.ylabel(iris.feature_names[pair[1]].title()[0:-4] + iris.feature_names[pair[1]][-4:])
    
    # Plot the training points for each species in turn
    for i, color, marker in zip(range(class_count), plot_colors, markers):
        # Subset the data to the class in question with the np.where() method
        index = np.where(y == i)
        # Plot the class in question on the subplot
        plt.scatter(X[index, 0], 
                    X[index, 1], 
                    c = color,
                    marker = marker,
                    label = iris.target_names[i],
                    cmap = plt.cm.brg, 
                    edgecolor = 'black', 
                    s = 15                       )

# Define a title for the overall collection of subplots after each subplot is fully defined
plt.suptitle('Decision Surface of a Decision Tree Using Paired Features',
             size = 24                                                   )

# Define the legend for the subplot collection
plt.legend(loc = 'lower right',
           fontsize = 16,
           borderpad = 0.1, 
           handletextpad = 0.1 )

# Set limits just large enough to show everything cleanly
plt.axis("tight")


# Note that in the above plots, there are occasionally generated small islands or peninsulae of classification which show us the Achilles' heel of Decision tree models: their tendency to overfit the data. While these small cutouts improve the accuracy of the model on the training data, they also result in a model that applies increasingly implausible critera to the data which likely won't stand up when new data is used to test the model. 
# 
# It is also interesting to note that with only two parameters available, a very high degree of accuracy is achievable in the last subplot using only petal length and petal, width with only a single modest and sensibly positioned peninsula.
# 
# We'll come back to this last subplot at the end of this posting.

# The very short and simple code below shows the results of a decision tree model when all four parameters are considered via the plot_tree() function.  As it turns out, only two parameters, petal length (X\[2\]) and petal width (X\[3\]), are needed to provide a very high degree of accuracy, yielding results very similar to what is achieved when considering only two parameters. The X\[1\] parameter for sepal width is also brought to bear to make the finest distinctions, but ultimately doesn't bring much value.

# In[ ]:


# Apply the decision tree classifier model to the data using all four parameters at once.
model_all_params = DecisionTreeClassifier().fit(iris.data, iris.target)
# Prepare a plot figure with set size.
plt.figure(figsize = (20,10))
# Plot the decision tree, showing the decisive values and the improvements in Gini impurity along the way.
plot_tree(model_all_params, 
          filled=True      )
# Display the tree plot figure.
plt.show()


# We may very simply add the max_depth parameter to the model with a value of 3 and observe the effect of forcing some simplicity on the tree, akin to pruning all the smallest branches of a tree down to a few points of divergence closest to the trunk.

# In[ ]:


# Apply the decision tree classifier model to the data using all four parameters at once, but with a maximum tree depth of 3
model_all_params_max_depth_3 = DecisionTreeClassifier(max_depth = 3).fit(iris.data, iris.target)
# Prepare a plot figure with set size.
plt.figure(figsize = (16,10))
# Plot the decision tree.
plot_tree(model_all_params_max_depth_3,
          rounded = True,
          filled = True                )
# Display the tree plot figure.
plt.show()


# This leaves (no pun intended) us with only four misclassified points by applying a total of four tests, and trimming one level sooner, i.e., setting max_depth to 2, would only have six errors out of the 150 points. 
# 
# Another approach we might consider is to mandate that there must be a significant increase in impurity for the split to occur.  Setting the min_impurity_decrease parameter will allow us to restrict splits to those that significantly improve the results of the overal model behavior.

# In[ ]:


# Apply the model to the data as before, but with a minimum impurity decrease of 0.01
model_all_params_min_imp_dec_001 = DecisionTreeClassifier(min_impurity_decrease = 0.01).fit(iris.data, iris.target)
# Prepare a plot figure with set size.
plt.figure(figsize = (16,10))
# Plot the decision tree.
plot_tree(model_all_params_min_imp_dec_001,
          rounded = True,
          filled = True                )
# Display the tree plot figure.
plt.show()


# This approach gets us to three misclassified points out of 150 with the same number of tests as before with *max_depth* = 3, but we are now able to apply the questions a bit more efficiently. 

# Having made this improvement using the full dataset, let's now circle back to that last subplot which also used only petal length and petal width and see how it shows in plot_tree().

# In[ ]:


# Prepare a plot figure with set size.
plt.figure(figsize = (20,12))
# Plot the decision tree, showing the decisive values and the improvements in Gini impurity along the way.
plot_tree(models[5],
          rounded = True,
          filled = True  )
# Display the tree plot figure.
plt.show()


# Seven conditionals to yield only one misclassified sample out of 150 using only the two most important features out of the four available.  Not bad at all, but is there a better way? 
# 
# In subsequent postings I'll be using other models on the iris dataset in an effort to compare and contrast the different approaches.
