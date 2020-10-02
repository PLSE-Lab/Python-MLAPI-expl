#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt


# **About dataset**
# 
# We have collected data about a set of patients, all of whom suffered from the same illness. During their course of treatment, each patient responded to one of 5 medications, Drug A, Drug B, Drug c, Drug x and y.
# 
# **GOAL:** 
# Build a model to find out which drug might be appropriate for a future patient with the same illness. The feature sets of this dataset are Age, Sex, Blood Pressure, and Cholesterol of patients, and the target is the drug that each patient responded to.
# 

# **Loading Data**

# In[ ]:


my_data=pd.read_csv("../input/drug200.csv")
my_data[0:5]


# **Size of the data**

# In[ ]:


my_data.shape


# **Pre-processing**

#  **We declare the following variables: **
#  

# In[ ]:


#Feature Matrix:
X=my_data[["Age","Sex","BP","Cholesterol","Na_to_K"]].values
X[0:5]


# **N.B:**
# 
# We have catigorical variables in our features like "Sex" and "BP"
# Unfortunately,Sklearn Decision tree don't handle with catigorical variables so we have to convert them into numerical  variables.

# In[ ]:


from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

X[0:5]


# In[ ]:


#Target Variable:
y=my_data["Drug"]
y[0:5]


# **Setting up the Decision Tree**

# We will be using train/test split on our decision tree.

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)


# In[ ]:


print("Shape of X_trainset: ",X_trainset.shape)
print("Shape of y_trainset: ",y_trainset.shape)


# In[ ]:


print("Shape of X_testset: ",X_testset.shape)
print("Shape of y_testset: ",y_testset.shape)


# **WE ARE GOOD**

# **Modeling**
# 
# We will first create an instance of the DecisionTreeClassifier called drugTree.

# In[ ]:


drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters


# In[ ]:


#We will fit the data with the training feature matrix X_trainset and training response vector y_trainset
drugTree.fit(X_trainset,y_trainset)


# **Prediction**
# 
# Let's make some predictions on the testing dataset and store it into a variable called predTree.

# In[ ]:


predTree=drugTree.predict(X_testset)


# In[ ]:


print("Predictions : ",predTree[0:5])
print("Actual Values : \n",y_testset[0:5])


# **EVALUATION**

# In[ ]:


print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))


# **Visualization**
# 
# Lets visualize the tree

# In[ ]:


from sklearn.externals.six import StringIO
#import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#with open("drugTree.dot", "w") as f:
    #f = tree.export_graphviz(, out_file=f)
import graphviz

data = export_graphviz(drugTree,out_file=None,filled=True,rounded=True,special_characters=True)
graph = graphviz.Source(data)
graph

