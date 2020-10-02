#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


iris = pd.read_csv('../input/Iris.csv')


# In[ ]:


iris.head()


# In[ ]:


iris.describe()


# Describing the Target variable

# In[ ]:


iris.Species.describe()


# Checking the data set dimensions

# In[ ]:


iris.shape


# Drop the column 'ID' as it is not used in our analysis

# 

# In[ ]:


irisdata = iris.drop(['Id'], axis = 1)


# Few visualizations

# In[ ]:


irisdata.plot(kind='hist', subplots=True, layout =(2,2))


# Visualize using boxplot to identify the outliers

# In[ ]:


irisdata.plot(kind='box', subplots=True, layout =(2,2))


# Model Building using Decision tree(Classification tree)

# In[ ]:


import sklearn.model_selection as ms


# Separating predictors and target variable

# In[ ]:


X = iris.iloc[:, 1:5]
y = iris.loc[:, 'Species']


# Split the data into train and test data in the ration 2:1

# In[ ]:


X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.33, random_state=42)


# Import required packages for decision tree

# In[ ]:


from sklearn import tree
from sklearn.tree import export_graphviz
from subprocess import check_call


# Fit the model using CART algorithm

# In[ ]:


clf = tree.DecisionTreeClassifier()
treemodel = clf.fit(X_train, y_train)


# In[ ]:


from IPython.display import Image as Img


# In[ ]:


with open("treeop.dot", 'w') as f:
     treeop = tree.export_graphviz(treemodel, out_file='treeop.dot', feature_names = X.columns, filled = True)

check_call(['dot','-Tpng','treeop.dot','-o','tree1.png'])
Img("tree1.png")


# 
