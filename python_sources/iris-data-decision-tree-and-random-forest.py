#!/usr/bin/env python
# coding: utf-8

# Its a Iris data set analysis using Decision Tree visualization and Random Forest. U need to have pydot installed in your machine

# In[ ]:


import pandas as pd
from sklearn import tree
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv('../input/Iris.csv',header=0)


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data['SepalLengthCm'] = data['SepalLengthCm'].astype(int)


# In[ ]:


data['SepalLengthCm'] = data['SepalLengthCm'].astype(int)
data['SepalWidthCm'] = data['SepalWidthCm'].astype(int)
data['PetalLengthCm'] = data['PetalLengthCm'].astype(int)
data['PetalWidthCm'] = data['PetalWidthCm'].astype(int)


# In[ ]:


##mapping the species
d = {'Iris-setosa' : 1, 'Iris-versicolor' : 2, 'Iris-virginica' : 3}
data['Species'] = data['Species'].map(d)


# In[ ]:


data['Species']=data['Species'].astype(int)


# In[ ]:


features = list(data.columns[1:5])
features


# In[ ]:


y = data["Species"]
x = data[features]
Tree = tree.DecisionTreeClassifier()
Tree = Tree.fit(x,y)
output = Tree.predict([5.4,2,4.5,1])
print (output)


# In[ ]:


# Import the random forest package
from sklearn.ensemble import RandomForestClassifier 

# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 100)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(x,y)

# Take the same decision trees and run it on the test data
output = forest.predict([5.4,2,4.5,1])

print (output)

