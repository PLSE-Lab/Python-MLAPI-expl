#!/usr/bin/env python
# coding: utf-8

# # Decision Tree regression 
# 
# Python | Decision Tree Regression using sklearn
# 
# Decision Tree is a decision-making tool that uses a flowchart-like tree structure or is a model of decisions and all of their possible results, including outcomes, input costs and utility.
# 
# Decision-tree algorithm falls under the category of supervised learning algorithms. It works for both continuous as well as categorical output variables.
# 
# The branches/edges represent the result of the node and the nodes have either:
# 
#     Conditions [Decision Nodes]
#     Result [End Nodes]
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Importing the libraries

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# # Importing the dataset

# In[ ]:


dataset = pd.read_csv('../input/decision_tree_dataset.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# # Fitting Decision Tree Regression to the dataset

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)


# # Predicting a new result

# In[ ]:



y_pred = regressor.predict([[6.5]])
y_pred


# # Visualising the Decision Tree Regression results (higher resolution)

# In[ ]:


X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

