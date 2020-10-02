#!/usr/bin/env python
# coding: utf-8

#  **Is salary depends on position or exprerience?**

# Hello floks,
# 
# Today I am trying to work on **Position_Salaries dataset** using Decision tree algorithm.
# 
# So,the dataset includes columns for **Position** with values ranging from Business Analyst, Junior Consultant to CEO, Level ranging from 1-10 and finally the **Salary** associated with each **position** ranging from **$45000** to **$100000**.
# 
# 
# 

# **Goal:-**
# 
# The problem statement is that the candidate with level 6.5 had a previous salary of 160000. In order to hire the candidate for a new role, the company would like to confirm if he is being honest about his last salary so it can make a hiring decision.
# 

# **Step 1-Data Prepocessing**

# In[ ]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[ ]:


# Importing the dataset
dataset = pd.read_csv("../input/Position_Salaries.csv")


# In[ ]:


#Checking the dataset
dataset


# ---As the dataset is too small we done required any kind of data preprocessing like dimensionality check ,null values check,replacement etc....

# **Plot between Level vs Salary**
# 
# Let's visualize the data before doing any modification on that!

# In[ ]:


a = dataset.iloc[:, 1].values
b = dataset.iloc[:, 2].values
plt.scatter(a,b,color='red',s=50)
plt.xlabel('Level')
plt.ylabel('Salary')
plt.title('Level vs salary')
plt.show()


# ---As per dataset as level or position increases ,salary also increase..

# In[ ]:


#Separating the dependent and independent variables
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# In[ ]:


print(X)


# In[ ]:


print(y)


# --- dataset is very small so, spliting dataset is not required!

# **Step 2-Building a model using decision tree algorithm.**

# In[ ]:


# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)


# **Prediction part**
# The problem statement is that the candidate with level 6.5 had a previous salary of 160000.
# 

# In[ ]:


y_pred = regressor.predict(np.array(6.5).reshape(-1,1))


# In[ ]:


print(y_pred)


# **Fun part**
# Visualize the result

# In[ ]:



# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'black')
plt.plot(X_grid, regressor.predict(X_grid), color = 'red')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# In[ ]:




