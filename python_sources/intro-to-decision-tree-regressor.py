#!/usr/bin/env python
# coding: utf-8

# This the notebook for Decision Tree Regression. In the first code segment, we import the necessary headers.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Next, input the dataset and define the independent and dependent variables

# In[ ]:


# Importing the dataset
dataset = pd.read_csv('../input/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values.reshape(-1, 1) ## had to add the reshape here to convert the array from 2d to 1d NOTE


# Import the model and fit it to the dataset

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(random_state=0)
reg.fit(X, y)


# Once trained, the model can be used to predict the value we want, but remember to feature scale the new value, while also making sure that it is a 2D array.

# In[ ]:


# Predicting a new result
"""y_pred = reg.predict(sc_X.transform([[6.5]]))
y_pred = sc_y.inverse_transform(y_pred)"""
y_pred = reg.predict([[6.5]])
y_pred


# Lastly, let's just visualize and see how well our DT just did.

# In[ ]:


# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, reg.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

