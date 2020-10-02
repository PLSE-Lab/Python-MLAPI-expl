#!/usr/bin/env python
# coding: utf-8

# <h1><center>Decision Trees for Binary Classification on Breast Cancer Wisconsin (Diagnostic) Data Set</center></h1>

# ## DataSet 
# Breast Cancer Wisconsin (Diagnostic) Data Set (https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)
# 
# ## Overview
# Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. n the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].
# 
# Attribute Information:
# 
# 1) ID number 2) Diagnosis (M = malignant, B = benign) 3-32)
# 
# Ten real-valued features are computed for each cell nucleus:
# 
# a) radius (mean of distances from center to points on the perimeter) b) texture (standard deviation of gray-scale values) c) perimeter d) area e) smoothness (local variation in radius lengths) f) compactness (perimeter^2 / area - 1.0) g) concavity (severity of concave portions of the contour) h) concave points (number of concave portions of the contour) i) symmetry j) fractal dimension ("coastline approximation" - 1)
# 
# The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.
# 
# All feature values are recoded with four significant digits.
# 
# ## Reference
# 1. https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
# 1. https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
# 1. https://www.coursera.org/learn/machine-learning-with-python/home/welcome
# 1. https://scikit-learn.org/stable/modules/tree.html
# 1. https://statinfer.com/204-3-10-pruning-a-decision-tree-in-python/
# 1. https://www.datacamp.com/community/tutorials/decision-tree-classification-python

# ## Task
# * Apply Decision Tree on the selected dataset
# * Apply two different heuristics for split (Entropy, Gini)
# * Apply pruning as well

# ## Import usefull libraries
# 
# * numpy (as np)
# * pandas
# * DecisionTreeClassifier from sklearn.tree

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics # for score / accuracy


# ## Load Dataset

# In[ ]:


#display input file path
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


#Load dataset
dataset = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv", delimiter=",")

# print dataset
dataset[:]


# ## Select Features

# In[ ]:


#print all columns (features and lable)
dataset.columns
# len(dataset.columns)


# In[ ]:


#Select Features 
X = dataset[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst','perimeter_worst', 'area_worst', 'smoothness_worst','compactness_worst','concavity_worst', 'concave points_worst','symmetry_worst', 'fractal_dimension_worst']].values
X [0:2]


# ## Lables
# 
# 1. **M** = malignant
# 1. **B** = benign

# In[ ]:


#display Unique Lable along with its count
dataset['diagnosis'].value_counts()


# In[ ]:


y = dataset['diagnosis'] 
y


# In[ ]:


# Split dataset into two part train and test dataset 70% training and 30% testset (Random)
from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)


# ## Apply First Heuristic (Entropy)

# In[ ]:


def MyDecisionTreeClassifier(heuristic, tree_depth = None):
    decision_tree_clfr = DecisionTreeClassifier(criterion = heuristic, max_depth = tree_depth)
    
    #Apply classifier on training dataset
    decision_tree_clfr.fit(X_trainset, y_trainset)
    
    return decision_tree_clfr


# In[ ]:


heuristic = "entropy"
decision_tree = MyDecisionTreeClassifier(heuristic)

# predict lables using remaining testset
predTree = decision_tree.predict(X_testset)


# ## Evaluation

# In[ ]:


print("Decision Trees's Accuracy using entropy: ", metrics.accuracy_score(y_testset, predTree))
print("Depth of Decision Tree: ", decision_tree.tree_.max_depth)


# ## Visualization
# Install below package for visualization
# 1. Install pydotplus 
# 1. Install python-graphviz

# In[ ]:


get_ipython().system('conda install -c conda-forge pydotplus -y')
get_ipython().system('conda install -c conda-forge python-graphviz -y')


# In[ ]:


import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
import numpy as np


# In[ ]:


fileName_entropy = "decision-tree-entropy.png"

dot_data = StringIO()
featureNames = dataset.columns[2:32]
labedNames = dataset["diagnosis"].unique().tolist()
    
# export_graphviz will convert decision tree classifier into dot file
tree.export_graphviz(decision_tree,feature_names = featureNames, out_file = dot_data,
                         class_names = np.unique(y_trainset), filled = True,  special_characters = True,rotate = False) 
    
# Convert dot file int pgn using pydotplus
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    
#write pgn into file
graph.write_png(fileName_entropy)

#display tree image
img_entropy = mpimg.imread(fileName_entropy)
plt.figure(figsize=(100, 200))
plt.imshow(img_entropy, interpolation='nearest')


# ## Apply Second Heuristic (Gini)

# In[ ]:


heuristic_g = "gini"
decision_tree_g = DecisionTreeClassifier(criterion = heuristic_g)

#Apply classifier on train dataset
decision_tree_g.fit(X_trainset,y_trainset)

# predict lables using remaining testset
predTree_g = decision_tree_g.predict(X_testset)


# ## Evaluation

# In[ ]:


print("Decision Trees's Accuracy using Gini: ", metrics.accuracy_score(y_testset, predTree_g))
print("Depth of Decision Tree: ", decision_tree_g.tree_.max_depth)


# ## Visualization

# In[ ]:


fileName_g = "decision-tree-gini.png"


dot_data = StringIO()
featureNames = dataset.columns[2:32]
labedNames = dataset["diagnosis"].unique().tolist()
    
# export_graphviz will convert decision tree classifier into dot file
tree.export_graphviz(decision_tree_g,feature_names = featureNames, out_file = dot_data,
                         class_names = np.unique(y_trainset), filled = True,  special_characters = True,rotate = False) 
    
# Convert dot file int pgn using pydotplus
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    
#write pgn into file
graph.write_png(fileName_g)

#display tree image
img_g = mpimg.imread(fileName_g)
plt.figure(figsize=(100, 200))
plt.imshow(img_g, interpolation='nearest')


# ## Pruning to Avoid Overfitting
# 
# How we can avoid Overfitting ?
# Overfitting can be avoided by using these parameters 
# * max_leaf_nodes
# * min_samples_leaf
# * max_depth
# 
# Description
# 1. max_leaf_nodes: This parameter can be used to define the max number of leaf nodes
# 1. min_samples_leaf: This parameter can be userd to restrict the size of sample leaf
# 1. max_depth: It can be used to reduce the depth of the tree to build a generalized tree

# In[ ]:


startingPoint = 2
accuracy_1 = np.zeros((decision_tree.tree_.max_depth - 1))

for x in range(startingPoint, decision_tree.tree_.max_depth + 1):
    heuristic = "entropy"
    decision_tree = MyDecisionTreeClassifier(heuristic)

    # predict lables using remaining testset
    predTree = decision_tree.predict(X_testset)
    
    accuracy_1 [x-startingPoint] = metrics.accuracy_score(y_testset, predTree)
    
    print("Decision Trees's Accuracy (entropy) with depth:", x , " is ", accuracy_1 [x-startingPoint],"")


# In[ ]:


import matplotlib.pyplot as plt
def ShowAccuracy(_range, data):
    plt.plot(range(2,_range+1),data,'g')
    plt.legend(('Accuracy'))
    plt.ylabel('Accuracy ')
    plt.xlabel('Depth')
    plt.tight_layout()
    plt.show()


# In[ ]:


ShowAccuracy(decision_tree.tree_.max_depth, accuracy_1)


# In[ ]:


startingPoint = 2
depth = np.zeros((decision_tree_g.tree_.max_depth - 1))
accuracy_2 = np.zeros((decision_tree_g.tree_.max_depth - 1))

for x in range(startingPoint, decision_tree_g.tree_.max_depth + 1):
    heuristic = "gini"
    decision_tree = MyDecisionTreeClassifier(heuristic)

    # predict lables using remaining testset
    predTree = decision_tree.predict(X_testset)
    
    depth [x-startingPoint] = x;
    accuracy_2 [x-startingPoint] = metrics.accuracy_score(y_testset, predTree)
    
    print("Decision Trees's Accuracy (Gini) with depth:",x, " is ", accuracy_2 [x-startingPoint],"")


# In[ ]:


ShowAccuracy(decision_tree_g.tree_.max_depth, accuracy_2)


# ### Pruning Results
# 
# * in case of first heuristic the best accuracy was 0.96 when tree depth was 4
# * In case of second heuristic the best accuracy was 0.95 when tre depth was 7

# ## Conclusion
