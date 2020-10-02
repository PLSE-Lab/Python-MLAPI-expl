#!/usr/bin/env python
# coding: utf-8

# # Understanding decision trees

# This kernel was inspired by the kernel - [Do You Have Spinal Disease? Decision Tree in R](https://www.kaggle.com/petrkajzar/do-you-have-spinal-disease-decision-tree-in-r)

# ## What are Decision Trees?
# >A decision tree is a flowchart-like structure in which each internal node represents a "test" on an attribute (e.g. whether a coin flip comes up heads or tails), each branch represents the outcome of the test, and each leaf node represents a class label (decision taken after computing all attributes).
# 
# *(Source: [Wikipedia](https://en.wikipedia.org/wiki/Decision_tree#Overview))*
# 
# In simpler terms, a decision tree checks if an attribute or a set of attributes satisfy a condition and based on the result of the check, the subsequent checks are performed. The tree splits the data into different parts based these checks.

# ## What is achieved in this kernel?
# The following are achieved in this dataset
# * Loading the data
# * Visualizing the data using a correlaton matrix and a pair plot
# * Building a Decision Tree Classifier
# * Determining the accuracy of the model using a confusion matrix
# * Viualizing the Decision tree as a flow chart

# ## Importing the necessary libraries

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tflearn.data_utils as du
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus


# In[ ]:


data = pd.read_csv('../input/column_3C_weka.csv')


# The dataset used here is the [Biomechanical features of orthopedic patients](https://www.kaggle.com/uciml/biomechanical-features-of-orthopedic-patients)

# In[ ]:


data.info()


# ## What is correlation?
# Correlation is a statistical term which in common usage refers to how close two variables are to having a linear relationship with each other. 
# 
# For example, two variable which are linearly dependent (say, x and y which depend on each other as x = 2y) will have a higher correlation than two variables which are non-linearly dependent (say, u and v which depend on each other as u = v<sup>2</sup>)

# In[ ]:


# Calculating the correlation matrix
corr = data.corr()
# Generating a heatmap
sns.heatmap(corr,xticklabels=corr.columns, yticklabels=corr.columns)


# In[ ]:


sns.pairplot(data)


# In the above two plots you can clearly see that the pairs of independent variables with a higher correlation have a more linear scatter plot than the independent variables having a relatively lesser correlation

# ## Splitting the dataset into independent (x) and dependent (y) variables

# In[ ]:


x = data.iloc[:,:6].values
y = data.iloc[:,6].values


# ## Splitting the dataset into train and test data
# The train data to train the model and the test data to validate the model's performance

# In[ ]:


x_train , x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# ## Scaling the independent variables
# [This](https://stackoverflow.com/questions/26225344/why-feature-scaling#26229427) question on stackoverflow has responses which gives a brief explanation on why scaling is necessary and how it can affect the model

# In[ ]:


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# ## Building the Decision tree
# The criterion here is `entropy`. The criterion parameter detemines the function to measure the quality of a split. When the `entropy` is used as a criterion, each split tries to reduce the randomness in that part of the data.
# 
# The another parameter used is the `max_depth`. This determines how deep a tree can go. The affect of this parameter on the model will be discusses later in this notebook

# In[ ]:


classifier = DecisionTreeClassifier(criterion = 'entropy', max_depth = 4)
classifier.fit(x_train, y_train)


# ## Making the prediction on the test data

# In[ ]:


y_pred = classifier.predict(x_test)


# ## What is a confusion matrix?
# >A confusion matrix is a technique for summarizing the performance of a classification algorithm.
# Classification accuracy alone can be misleading if you have an unequal number of observations in each class or if you have more than two classes in your dataset.
# Calculating a confusion matrix can give you a better idea of what your classification model is getting right and what types of errors it is making.
# 
# *[Source](https://machinelearningmastery.com/confusion-matrix-machine-learning/)*

# In[ ]:


cm = confusion_matrix(y_test, y_pred)


# In[ ]:


accuracy = sum(cm[i][i] for i in range(3)) / y_test.shape[0]
print("accuracy = " + str(accuracy))


# ## Visualizing the Decision Tree

# In[ ]:


dot_data = StringIO()

export_graphviz(classifier, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# ## Building a model without the `max_depth` parameter

# In[ ]:


classifier2 = DecisionTreeClassifier(criterion = 'entropy')
classifier2.fit(x_train, y_train)


# In[ ]:


y_pred2 = classifier2.predict(x_test)


# In[ ]:


cm2 = confusion_matrix(y_test, y_pred2)


# In[ ]:


accuracy2 = sum(cm2[i][i] for i in range(3)) / y_test.shape[0]
print("accuracy = " + str(accuracy2))


# ## Visualizing the decision tree without the `max_depth` parameter

# In[ ]:


dot_data = StringIO()

export_graphviz(classifier2, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# Now, consider the leaf nodes (terminal nodes) of the tree with and without the `max_depth` paratemer. You will notice that the entropy all the terminal nodes are zero in the tree without the `max_depth` parameter and non zero in three with that parameter. This is because when the parameter is not mentioned, the split recursively takes place till the terminal node has an entropy of zero.

# To know more about the different paramteres of the `sklearn.tree.DecisionTreeClassifier`, click [here](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
