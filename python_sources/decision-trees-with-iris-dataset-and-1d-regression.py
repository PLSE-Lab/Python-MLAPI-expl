#!/usr/bin/env python
# coding: utf-8

# ## Last Updated:12.5.20
# # Please UPVOTE!! if you Liked this notebook.

# # Decision Trees
#  Decision Trees are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

# ## Some advantages of decision trees are:
# 
# * Simple to understand and to interpret. Trees can be visualised.
#  
# * Requires little data preparation. Other techniques often require data normalisation, dummy variables need to be created and blank values to be removed. Note however that this module does not support missing values.
# 
# * The cost of using the tree (i.e., predicting data) is logarithmic in the number of data points used to train the tree.
# 
# * Able to handle both numerical and categorical data. Other techniques are usually specialised in analysing datasets that have only one type of variable. See algorithms for more information.
# 
# * Able to handle multi-output problems.
# 
# * Uses a white box model. If a given situation is observable in a model, the explanation for the condition is easily explained by boolean logic. By contrast, in a black box model (e.g., in an artificial neural network), results may be more difficult to interpret.
# 
# * Possible to validate a model using statistical tests. That makes it possible to account for the reliability of the model.
# 
# * Performs well even if its assumptions are somewhat violated by the true model from which the data were generated.
# 
# ## The disadvantages of decision trees include:
# 
# * Decision-tree learners can create over-complex trees that do not generalise the data well. This is called overfitting. Mechanisms such as pruning, setting the minimum number of samples required at a leaf node or setting the maximum depth of the tree are necessary to avoid this problem.
# 
# * Decision trees can be unstable because small variations in the data might result in a completely different tree being generated. This problem is mitigated by using decision trees within an ensemble.
# 
# * The problem of learning an optimal decision tree is known to be NP-complete under several aspects of optimality and even for simple concepts. Consequently, practical decision-tree learning algorithms are based on heuristic algorithms such as the greedy algorithm where locally optimal decisions are made at each node. Such algorithms cannot guarantee to return the globally optimal decision tree. This can be mitigated by training multiple trees in an ensemble learner, where the features and samples are randomly sampled with replacement.
# 
# * There are concepts that are hard to learn because decision trees do not express them easily, such as XOR, parity or multiplexer problems.
# 
# * Decision tree learners create biased trees if some classes dominate. It is therefore recommended to balance the dataset prior to fitting with the decision tree.

# # Decision Trees with IRIS Dataset

# Plot the decision surface of a decision tree on the iris dataset
# Plot the decision surface of a decision tree trained on pairs of features of the iris dataset.
# 
# See decision tree <tree> for more information on the estimator.
# 
# For each pair of iris features, the decision tree learns decision boundaries made of combinations of simple thresholding rules inferred from the training samples.
# 
# We also show the tree structure of a model built on all of the features.

# ![](https://evolution.berkeley.edu/evolibrary/images/interviews/flower_diagram.gif)
# ![](https://cdn-images-1.medium.com/max/2000/1*7bnLKsChXq94QjtAiRn40w.png)

# In[ ]:


from sklearn.datasets import load_iris #import dataset
iris = load_iris()                     #loading the data
iris                                   #print the data


# In[ ]:


get_ipython().system('pip install pydotplus')


# In[ ]:


from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus
from IPython.display import Image

clf = DecisionTreeClassifier().fit(iris.data, iris.target)
dot_data = export_graphviz(clf, out_file=None, filled=True, rounded=True,
                                feature_names=iris.feature_names,  
                                class_names=['Versicolor','Setosa','Virginica'])
graph = pydotplus.graph_from_dot_data(dot_data)  
image=graph.create_png()
graph.write_png("kmc_dt.png")
Image(filename="kmc_dt.png")


# # Decision Tree Regression
# A 1D regression with decision tree.
# 
# The decision trees <tree> is used to fit a sine curve with addition noisy observation. As a result, it learns local linear regressions approximating the sine curve.
# 
# We can see that if the maximum depth of the tree (controlled by the max_depth parameter) is set too high, the decision trees learn too fine details of the training data and learn from the noise, i.e. they overfit.

# In[ ]:


print(__doc__)

# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Plot the results
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black",
            c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue",
         label="max_depth=2", linewidth=1)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()


# # If you have came till here don't forget to UPVOTE!!keep me motivated!!:)
