#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd

# SciKit Learn
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import cross_val_score, train_test_split

# For visualizing the tree
from graphviz import Source
from IPython.display import SVG


# # Step 1. Read the data into a Pandas DataFrame

# In[ ]:


data = pd.read_csv("../input/Iris.csv", index_col="Id")
data.head()


# # Step 2. Extract the data into features (inputs) and targets (outputs) and create a train/test split

# In[ ]:


iris_features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
iris_targets = "Species"
iris_train, iris_test = train_test_split(data, test_size=0.2) # We want a 80%/20% split for training/testing
print(f"Train: {len(iris_train)} rows\nTest: {len(iris_test)} rows")


# # Step 3. Create the Decision Tree Classifier and fit it to the data
# We want a 80%/20% split for training/testing

# In[ ]:


iris_classifier = DecisionTreeClassifier(random_state=0)
model = iris_classifier.fit(X=iris_train[iris_features], y=iris_train[iris_targets])
print(model)


# # Step 4. Determine our the model's accuracy with the test data

# In[ ]:


model.score(X=iris_test[iris_features], y=iris_test[iris_targets])


# Over 90% -- not too bad!

# # Let's Take a look at the decision tree

# In[ ]:


graph = Source(export_graphviz(model, out_file=None, feature_names=iris_features, filled=True, class_names=model.classes_))
SVG(graph.pipe(format='svg'))


# ## At this point, we can stop and be happy with our tree. But can we do better?
# If we try to build the tree using a plethora of random states, we'll find that some output better trees than others. 
# 
# Let's try to optimize the tree using a method called Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# # Step 5. Build 100 random Decision Trees using the RandomForestClassifier

# In[ ]:


forest = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=2)


# # Step 6. Fit the trees in the forest, and determine the 'best' one.
# In this case, we're going to look for the tree with the lowest node count (ie the smallest tree).

# In[ ]:


forest_model = forest.fit(X=iris_train[iris_features], y=iris_train[iris_targets])
size, index = min((estimator.tree_.node_count, idx) for (idx,estimator) in enumerate(forest.estimators_))
print(f'The smallest tree has {size} nodes!')


# Wow, that's much smaller than the tree we made above!

# In[ ]:


smallest_tree = forest_model.estimators_[index]
smallest_tree = smallest_tree.fit(X=iris_train[iris_features], y=iris_train[iris_targets])
smallest_tree.score(X=iris_test[iris_features], y=iris_test[iris_targets])


# So this tree has a mere five nodes but still remains over 90% accurate. Let's see what is looks like

# In[ ]:


graph = Source(export_graphviz(smallest_tree, out_file=None, feature_names=iris_features, filled=True, class_names=model.classes_))
SVG(graph.pipe(format='svg'))


# 

# **As you can see, this tree performs equally well with much fewer decisions. This tells us a few things, for example: most of the features are irrelevant for this prediction!**
