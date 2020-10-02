#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ### Introduction
# 
# Decision Trees are classification methods that are able to extract simple rules about the data features which are inferred from the input
# dataset. Several algorithms for decision tree induction are available in the literature. Scikit-learn contains the implementation of the CART (Classification and Regression Trees) induction algorithm.
# 
# First, we use make_classificaton to create an artificial classification dataset. This convenience function allows fine-grained control over the characteristics of the dataset it produces. We create a dataset with 1,000 instances. Of the 100 features, 20 are informative; the remainder are redundant combinations of the information features, or noise. We then train and evaluate a single decision tree, followed by a random forest with 10 trees. The random forest's F1 precision, recall, and F1 scores are greater.

# In[ ]:


import pandas as pd
import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#Setting random seed
seed = 10
X, y = make_classification(
n_samples=1000, n_features=100, n_informative=20,
n_clusters_per_class=2,
random_state=11)
X_train, X_test, y_train, y_test = train_test_split(X, y,
random_state=11)
clf = DecisionTreeClassifier(random_state=11)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(classification_report(y_test, predictions))


# ### Decison Tree Application on Iris Dataset
# 
# Here, we apply decision tree classifier on Iris dataset. First, we import all the libraries needed for this example. Scikit-learn does not implement any post-prunning step. So, to avoid overfitting, we can control the tree size with the parameters min_samples_leaf, min_samples_split and max_depth. 

# In[ ]:


# Decision Tree on Iris Dataset

import pandas as pd
import graphviz
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Setting random seed.
seed = 10

# Loading Iris dataset.
data = pd.read_csv('../input/iris/Iris.csv')
print(data.head())
# Creating a LabelEncoder and fitting it to the dataset labels.
le = LabelEncoder()
le.fit(data['Species'].values)
# Converting dataset str labels to int labels.
y = le.transform(data['Species'].values)
# Extracting the instances data.
X = data.drop('Species', axis=1).values
# Splitting into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, stratify=y, random_state=seed)

# Creating a DecisionTreeClassifier.
# The criterion parameter indicates the measure used (possible values: 'gini' for the Gini index and
# 'entropy' for the information gain).
# The min_samples_leaf parameter indicates the minimum of objects required at a leaf node.
# The min_samples_split parameter indicates the minimum number of objects required to split an internal node.
# The max_depth parameter controls the maximum tree depth. Setting this parameter to None will grow the
# tree until all leaves are pure or until all leaves contain less than min_samples_split samples.
tree = DecisionTreeClassifier(criterion='gini',
min_samples_leaf=5,
min_samples_split=5,
max_depth=None,
random_state=seed)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print('DecisionTreeClassifier accuracy score: {}'.format(accuracy))

def plot_tree(tree, dataframe, label_col, label_encoder, plot_title):
    label_names = pd.unique(dataframe[label_col])
    # Obtaining plot data.
    graph_data = export_graphviz(tree, feature_names=dataframe.drop(label_col, axis=1).columns,
    class_names=label_names,filled=True,rounded=True, out_file=None)
    # Generating plot.
    graph = graphviz.Source(graph_data)
    graph.render(plot_title)
    return graph

tree_graph = plot_tree(tree, data, 'Species', le, 'Iris')
tree_graph


# ### Dataset with Categorical Features
# 
# Unfortunately, the DecisionTreeClassifier class does not handle categorical features directly. So, we might consider to transform them to
# dummy variables. However, this approach must be taken with a grain of salt because decision trees tend to overfit on data
# with a large number of features.
# 
# Here we build two trees with different depth to analyze overfitting, and deal with overfitting by building a smaller tree. We can observe that the second tree is almost as accurate as the first one. Apparently both trees are able to handle the mushroom data pretty well. The second three might be preferred, since it is a simpler and computationally cheaper model. Finally, we plot the second tree.

# In[ ]:


import pandas as pd
import graphviz
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Setting random seed.
seed = 10

# Loading Mushroom dataset.
data = pd.read_csv('../input/mushroom-classification/mushrooms.csv')
# We drop the 'stalk-root' feature because it is the only one containing missing values.
data = data.drop('stalk-root', axis=1)
# Creating a new DataFrame representation for each feature as dummy variables.
dummies = [pd.get_dummies(data[c]) for c in data.drop('class', axis=1).columns]
# Concatenating all DataFrames containing dummy variables.
binary_data = pd.concat(dummies, axis=1)
# Getting binary_data as a numpy.array.
X = binary_data.values
# Getting the labels.
le = LabelEncoder()
y = le.fit_transform(data['class'].values)
# Splitting the binary dataset into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, stratify=y, random_state=seed)

# Creating a DecisionTreeClassifier.
tree = DecisionTreeClassifier(criterion='gini', min_samples_leaf=5, min_samples_split=5, max_depth=None,
random_state=seed)
tree.fit(X_train, y_train)

# Prediction and Accuracy
y_pred = tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('DecisionTreeClassifier accuracy score: {}'.format(accuracy))

print('DecisionTreeClassifier max_depth: {}'.format(tree.tree_.max_depth))

#What if we fit a decision tree with a smaller depth?
tree = DecisionTreeClassifier(criterion='gini',
min_samples_leaf=5,
min_samples_split=5,
max_depth=3,
random_state=seed)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('DecisionTreeClassifier accuracy score: {}'.format(accuracy))

# Appending 'label' column to binary DataFrame.
binary_data['class'] = data['class']
tree_graph = plot_tree(tree, binary_data, 'class', le, 'Mushroom')
tree_graph

# Feature Importance
print("Number of Features :", tree.n_features_,", number of classes :\n",tree.n_classes_)
print("Feature Importance :\n",tree.feature_importances_)


# ### Bagging
# 
# Bootstrap aggregating, or bagging, is an ensemble meta-algorithm that can reduce the variance in an estimator. Bagging can be used in classification and regression tasks. When the component estimators are regressors, the ensemble averages their predictions. When the component estimators are classifiers, the ensemble returns the mode class.
# 
# Bagging independently fits multiple models on variants of the training data. The training data variants are created using a procedure called bootstrap resampling. Often it is necessary to estimate a parameter of an unknown probability distribution using only a sample of the distribution. 
# 
# Bagging is a useful meta-algorithm for estimators that have high variance and low bias, such as decision trees. In fact, bagged decision tree ensembles are used so often and successfully that the combination has its own name: the random forest. 
# 
# The number of trees in the forest is an important hyperparameter. Increasing the number of trees improves the model's performance at the cost of computational complexity.
# 
# Regularization techniques, such as pruning or requiring a minimum number of training instances per leaf node, are less important when training trees for forests than they are for training a single estimator, as bagging provides regularization. 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#Setting random seed
seed = 10

# Dataset Creation
X, y = make_classification(n_samples=1000, n_features=100, n_informative=20,
n_clusters_per_class=2,random_state=11)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=11)

clf = RandomForestClassifier(n_estimators=10, random_state=11)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(classification_report(y_test, predictions))

# Feature Importance
print("Number of Features :", clf.n_features_,", number of classes :\n",clf.n_classes_)
print("Feature Importance :\n",clf.feature_importances_)


# ### Boosting
# 
# Boosting is a family of ensemble methods that are primarily used to reduce the bias of an estimator. Boosting can be used in classification and regression tasks. Like bagging, boosting creates ensembles of homogeneous estimators. 
# 
# We will focus our discussion of boosting on one of the most popular boosting
# algorithms, AdaBoost. AdaBoost is an iterative algorithm that was formulated by Yoav Freund and Robert Schapire in 1995. It's name is a portmanteau of adaptive boosting. 
# 
# On the first iteration, AdaBoost assigns equal weights to all
# of the training instances and then trains a weak learner. A weak learner (or weak classifier, weak predictor, and so on), is defined only as an estimator that performs slightly better than random chance, such as a decision tree with one or a small number of nodes. Weak learners are often, but not necessarily,
# simple models. A strong learner, in contrast, is defined as an estimator that is arbitrarily better than a weak learner. 
# 
# Most boosting algorithms, including AdaBoost, can use any base estimator as a weak
# learner. On subsequent iterations, AdaBoost increases the weights of training instances that the previous iteration's weak learner predicted incorrectly and decreases the weights of the instances that were predicted correctly. It then trains another weak learner on the re-weighted instances. 
# 
# Subsequent learners increasingly focus on instances that the ensemble predicts incorrectly. The algorithm terminates when it achieves perfect performance, or after a specified number of iterations. The ensemble predicts the weighted sum of the base estimators' predictions.
# 
# scikit-learn implements a variety of boosting meta-estimators for classification and regression tasks, including AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, and GradientBoostingRegressor. 
# 
# In the following example, we train an AdaBoostClassifier for an artificial dataset created using the make_classification convenience function. We then plot the accuracy of the ensemble as the number of base estimators increases. We compare the ensemble's accuracy with the accuracy of a single decision tree

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Dataset creation
X, y = make_classification(
n_samples=1000, n_features=50, n_informative=30,
n_clusters_per_class=3,
random_state=11)

#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11)

#Model Creation
tree_clf = DecisionTreeClassifier(random_state=11)
tree_clf.fit(X_train, y_train)
print('Decision tree accuracy: %s' % tree_clf.score(X_test, y_test))

# When an argument for the base_estimator parameter is not passed, the default DecisionTreeClassifier is used
clf = AdaBoostClassifier(n_estimators=50, random_state=11)
clf.fit(X_train, y_train)
accuracies=[]
accuracies.append(clf.score(X_test, y_test))
plt.title('Ensemble Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of base estimators in ensemble')
plt.plot(range(1, 51), [accuracy for accuracy in clf.staged_score(X_test, y_test)])

# Feature Importance
print("Number of Features :", tree_clf.n_features_,", number of classes :\n",tree_clf.n_classes_)
print("Feature Importance :\n",tree_clf.feature_importances_)


# ### Conclusion
# 
# This is an indepth tutorial covering basic decision tree classifier, bagging, and boosting techniques. It shows you how you can print decision tree for visual analysis. Another important application of decision tree is feature selection which is done by printing feature importances calculated using decision trees. 
# 
# ### Note: 
# Please don't forget to like, share the tutorial and provide useful feedback in comments.
