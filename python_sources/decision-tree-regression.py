#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


# Input data files are available in the "../input/" directory
# print all the file/directories present at the path
import os
print(os.listdir("../input/"))


# In[ ]:


# importing the dataset
dataset = pd.read_csv('../input/Position_Salaries.csv')


# In[ ]:


dataset.head()


# In[ ]:


dataset.info()


# In[ ]:


dataset.isnull().sum()


# In[ ]:


plt.plot(dataset.iloc[:,1:-1],dataset.iloc[:,-1],color='red')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('Position Level VS Salary')
plt.show()


# In[ ]:


# matrix of features as X and dep variable as Y (convert dataframe to numpy array)
X = dataset.iloc[:,1:-1].values          #Level
Y = dataset.iloc[:,-1].values           #Salary


# In[ ]:


# Applying Desicion Tree Regressor

from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor()
reg.fit(X,Y)


# In[ ]:


# Predicting a new result
y_pred = reg.predict([[6.5]])


# In[ ]:


y_pred


# In[ ]:


X


# In[ ]:


# Visualising the Decision Tree Regression results (higher resolution)

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, reg.predict(X_grid), color = 'blue')
plt.title('Decision Tree Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

