#!/usr/bin/env python
# coding: utf-8

# ## Content
# 1. Introduction
# 2. Datasets
# 3. CART algorithm
# 4. Decision Tree model creation process theory
# 5. Misclassification method theory
# 6. Dataset overview
# 7. Misclassification Tree implementation
# 8. Conclusion
# ***

# ## 1. Introduction
# A [Decision Tree](https://en.wikipedia.org/wiki/Decision_tree) is a tree-like [model](https://en.wikipedia.org/wiki/Causal_model) that uses an intelligent [decision support system](https://en.wikipedia.org/wiki/Decision_support_system). With it,  a strategy can be identified to reach a goal. Also, it can be seen as a [flowchart](https://en.wikipedia.org/wiki/Flowchart)-like structure in which each node represents a "test" on an attribute. Each branch represents the outcome of the test, and each leaf node represents a class label. The paths from root to leaf represent classification rules. Decision trees are a valuable tool in machine learning.
# ***

# ## 2. Datasets
# It is recommended to use decision trees on datasets that have only **non-metric** data. We can think of each attribute of a dataset as being a question and the links connecting two nodes being an answer to that question. These methods work perfectly on small datasets. Moreover, decision trees work perfectly on small datasets.
# ***

# ## 3. CART algorithm
# **CART**([Classification and Regression Trees](https://machinelearningmastery.com/classification-and-regression-trees-for-machine-learning/), Breiman 1984) algorithm builds the tree using the train set. The dataset is split into: 
# 1. Train dataset, wich is used to build the tree;
# 2. Test dataset, wich is used to test the accuracy.
# 
# Every outcome of a node is called **split** and it coresponds to a training set.
# ***

# ## 4. Decision Tree model creation process theory
# First of all, the root node must be found. We do that by measuring the [entropy](http://people.revoledu.com/kardi/tutorial/DecisionTree/how-to-measure-impurity.htm) and split all the attributes. After that, we choose as root the attribute with the minimum split. We do the same thing recursively for each branch of root. A node cannot have children that are his ancestors. So, if a node ,,n'' is of type A, its children cannot be of type A, or of any type that its ancestors are.
# ***

# ## 5. Misclassification method theory
# This method measures the minimum probability that an element, from the training set, is misclassified by using the attribute A in node N.
# 
# It is defined by the following formula:
# ![formula](https://image.ibb.co/dhjftc/mie.png)
# 
# The node N has n elements. If we divide N using attribute A we obtain: 
# *  N_d, that has d elements;
# * N_s, that has s elements.
# 
# It's split index is:
# 
# ![](https://image.ibb.co/cxU5tc/mts.png)
# We split each permitted attribute. The minimum split gives us the desired attribute.
# ***

# ## 6. Dataset overview
# The following python implementation uses a database that contains classificated species. It consists of 15 entries. The attributes are: 
# * body temperature;
# * skin cover;
# * gives birth;
# * aquatic creature;
# * aerial creature;
# * has legs;
# * hibernates.
# 
# The classes are:
# * mammal;
# * reptile;
# * fish;
# * amphibian;
# * bird.
# 
# Let's take a look at the data distribution.

# In[ ]:


import queue
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

csv_dataset = pd.read_csv("../input/species_classification.csv")
csv_dataset.describe()


# In[ ]:


csv_dataset.plot.hist(alpha=0.25)
plt.show()


# ***

# ## 7. Misclassification Tree implementation
# First step is to create a node class. 

# In[ ]:


class MisclassificationTreeNode:
    parent = None
    branch = None
    attr_index = None
    permitted_attributes = None
    permitted_entries = None
    children = None
    classes_dist_dict = None
    class_label = None

    def __init__(self, parent=None,
                 attr_index=None,
                 children=None,
                 permitted_attributes=None,
                 permitted_entries=None,
                 classes_dist_dict=None,
                 branch=None,
                 class_type=None):
        self.class_label = class_type
        self.parent = parent
        self.attr_index = attr_index
        self.children = children
        self.permitted_attributes = permitted_attributes
        self.permitted_entries = permitted_entries
        self.classes_dist_dict = classes_dist_dict
        self.branch = branch


# Class overview:
# * parent, holds the parrent node;
# * branch, holds the current node branch; 
# * attr_index, holds the current node attribute index. If it is class it holds the class row index in db;
# * permitted_attributes, holds a list with the attributes indexes that can be used to create his children; 
# * permitted_entries, holds a list with the entries indexes that can be used to create his children;
# * children, holds a list with the current node children; 
# * classes_dist_dict, holds a dictionary with the current node attribute distribution; 
# * class_type, holds the dominant class type if it is a leaf (final node). 
#  
#  We need to define some methods that will help us create a misclassification tree.
#  
#  **Attr_distribution** method solves the attribute class distribution problem. It returns a dictionary. It has as keys the attribute index, and as values the attribute branch distribution dictionary. These dictionaries have as key the classes, and as values the branch distribution over an attribute.

# In[ ]:


def attr_distribution(x, y, distinct_classes, attr_index, permitted_entries_indexes):
    branches_class_distribution = {}
    branches = []
    for entry in permitted_entries_indexes:
        if x[entry][attr_index] not in branches:
            branches.append(x[entry][attr_index])
    for branch in branches:
        class_distribution_over_a_branch = {}
        for i in distinct_classes:
            class_distribution_over_a_branch.update({i: 0})
        total_of_a_single_branch = 0
        for permitted_entry_index in permitted_entries_indexes:
            if branch == x[permitted_entry_index][attr_index]:
                total_of_a_single_branch += 1
                class_distribution_over_a_branch[y[permitted_entry_index]] =                     class_distribution_over_a_branch[(
                        y[permitted_entry_index])] + 1
        branches_class_distribution.update({branch: class_distribution_over_a_branch})
    return branches_class_distribution


# **Split_attr** method returns the split value and a dictionary with the attribute distribution. 

# In[ ]:


def split_attr(x, y, distinct_classes, attr_index, permitted_entries_indexes):
    attr_dist = attr_distribution(x, y, distinct_classes, attr_index, permitted_entries_indexes)
    attr_split = 0
    for dictionary in attr_dist.values():
        attr_split += (1 - (max(dictionary.values()) / sum(dictionary.values()))) * (
                sum(dictionary.values()) / len(permitted_entries_indexes))
    return attr_split, attr_dist


# **Activate_node** method solves the following problems:
# * the permitted entries list, by creating the new list of permitted entries;
# * the permitted attributes list, by creating the new list of permitted attributes;
# * the node attribute, by splitting all the permitted attributes and choosing the one with the minimum split index;
# * it saves the attribute class distribution;
# * the node children, generates the node.
# 
# Each newly created child knows the following:
# * Its hierarchical parent;
# * The branch index (the link) that he is on (for example: if attribute sex has two genders: male and female, the male attribute is considered to be branch 0 and the female attribute to be branch 1);
# * if it is or not a leaf.
# 
# If one child is a leaf, the attribute field will be assigned with the class index, and the field of class type will hold the class label.

# In[ ]:


def activate_node(x, y, distinct_classes, class_column, node):
    entries = []
    attributes = []
    if node.parent is not None:
        for entry in node.parent.permitted_entries:
            if x[entry][node.parent.attr_index] == node.branch:
                entries.append(entry)
        for attr in node.parent.permitted_attributes:
            attributes.append(attr)
    else:
        for i in range(0, len(x)):
            entries.append(i)
        for i in range(0, len(x[0])):
            attributes.append(i)
    node.permitted_entries = entries
    attributes_split_index_dictionary = {}
    attributes_dist_dict = {}
    for attribute in attributes:
        split_index, attr_dist = split_attr(x, y, distinct_classes, attribute, entries)
        attributes_split_index_dictionary.update({attribute: split_index})
        attributes_dist_dict.update({attribute: attr_dist})
    attr_index = min(attributes_split_index_dictionary,
                        key=attributes_split_index_dictionary.get)
    attributes.remove(attr_index)
    children = []
    node.attr_index = attr_index
    node.permitted_attributes = attributes
    node.classes_dist_dict = attributes_dist_dict.get(attr_index)
    for key, value in attributes_dist_dict.get(attr_index).items():
        child = MisclassificationTreeNode(parent=node, branch=key)
        child.classes_dist_dict = value
        if sum(value.values()) == max(value) or len(attributes) == 0:
            child.attr_index = class_column
            for key_1, value_1 in value.items():
                if value_1 == max(value.values()):
                    child.class_type = key
        children.append(child)
    node.children = children


# First of all, we need to:
# 1. shuffle the database;
# 2. split the data intro train and test.

# In[ ]:


class_column = 7
db = csv_dataset.iloc[:, :].values.astype(np.int_)
np.random.shuffle(db)
y = db[:, class_column]
distinct_classes = np.unique(y)
x = np.delete(db, [class_column], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)


# The root is the first node to be activated. We must specify the permitted entries and attributes. It will use them to find the root attribute. For the root node, all the entries and all the attributes are permitted. In the following section, the root node has been created first. After that, we used a queue to create the rest of the tree.

# In[ ]:


root = MisclassificationTreeNode(parent=None)
nodes_to_activate = queue.Queue()
nodes_to_activate.put(root)
while not nodes_to_activate.empty():
    candidate = nodes_to_activate.get()
    activate_node(x_train, y_train, distinct_classes, class_column, candidate)
    for node in candidate.children:
        if node.attr_index is None:
            nodes_to_activate.put(node)


# Now that we have the tree, let's see how many nodes and leaves it has. Using a [BFS](https://en.wikipedia.org/wiki/Breadth-first_search) algorithm, we will count them.

# In[ ]:


q = queue.Queue()
q.put(root)
count_nodes = 1
count_leafs = 0
while not q.empty():
    candidate = q.get()
    count_nodes += 1
    if candidate.attr_index is class_column:
        count_leafs += 1
    if candidate.children is not None:
        for node in candidate.children:
            q.put(node)
print("The tree has %s nodes and %s leafs." % (count_nodes, count_leafs))


# In the following section, the testing set will be used to measure the tree accuracy. The creation process depends on how the database was shuffled. For that, we will create a method that finds the class of a new entry. **There will be cases where the tree cannot fi an answer.**

# In[ ]:


def decide(root, class_column, candidate):
    start = root
    while class_column is not start.attr_index:
        if start.attr_index == class_column:
            return start.class_type
        if candidate[start.attr_index] >= len(start.children):
            return -1
        else:
            start = start.children[candidate[start.attr_index]]
    return start.class_label


# In[ ]:


test_results = []
for entry in x_test:
    test_results.append(decide(root, class_column, entry))


# The tree works only on some king of entries. If an node must take a path that it does not exist the tree can't give a valid answer.
# ***

# ## 8. Conclusion
# 
# Decision trees are a valuable tool in the machine learning universe. Also, it is not as powerful as a [Neural Network](https://www.kaggle.com/andreicosma/back-propagation-neural-network). This method works good on small databses, but it may result an inconsistent learning rate. This is due to the lack of branches from a node.
# 
# This algorithm can build good and simple decision rules. The rules can be used to reach a certain goal, to classificate an object.
# 
# I hope this notebook helped you. 
# 
# References: Smaranda B. - Classification course.
# 
