#!/usr/bin/env python
# coding: utf-8

# This is a kernel for mobile price classification using scikit-learn decision tree.

# In[ ]:


import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn import tree
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO
import pydot


# # **Load data**

# In[ ]:


trainData = pd.read_csv('../input/train.csv')


# In[ ]:


trainData.head()


# # **Split attributes from the target class**

# In[ ]:


trainPriceRange = trainData["price_range"]
trainData = trainData.drop("price_range", axis=1)


# # **Run 10-k fold cross validation**

# In[ ]:


clf = tree.DecisionTreeClassifier(random_state=0)
cross_return = cross_validate(clf, trainData, trainPriceRange, cv=10, return_estimator=True)


# In[ ]:


cross_return["test_score"]


# In[ ]:


cross_return["test_score"].mean()


# # **Get the best tree from cross validation**

# In[ ]:


min_test = min(cross_return["test_score"])

max_test = max(cross_return["test_score"])
max_index = [i for i, j in enumerate(cross_return["test_score"]) if j == max_test]


# In[ ]:


best_tree = cross_return["estimator"][max_index[0]]


# # **Show feature importances**

# In[ ]:


best_tree.feature_importances_


# For this tree, the most relevant feature is the attribute 'ram', followed by 'battery_power' and size of the screen.

# In[ ]:


height = best_tree.feature_importances_
bars = trainData.columns
y_pos = np.arange(len(bars))
 
plt.bar(y_pos, height)
plt.xticks(y_pos, bars, rotation = 90)
plt.show()

