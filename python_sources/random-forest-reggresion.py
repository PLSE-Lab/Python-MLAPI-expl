#!/usr/bin/env python
# coding: utf-8

# importing library
# *  NumPy is a package in Python used for Scientific Computing. The ndarray (NumPy Array) is a multidimensional array used to store values of same datatype
# * matplotlib.pyplot for data visualization 
# * pandas for data manipulation  
# 

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# importing dataset to x and y

# In[ ]:


dataset = pd.read_csv('../input/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values


# Fitting the random_forest_reggresion Model to the dataset
# 

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
regg=RandomForestRegressor(n_estimators=300,random_state=0)
regg.fit(X,y)


# Predicting a new result
# 

# In[ ]:


y_pred = regg.predict([[6.5]])
print(y_pred)


# Visualising the Regression results (for higher resolution and smoother curve)
# 

# In[ ]:


X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regg.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (random forest regg Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

