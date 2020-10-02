#!/usr/bin/env python
# coding: utf-8

# ## Linear Regression (**Machine Learning**)

# ### Simple example how Linear Regression works

# ![](https://preview.ibb.co/h86Jta/linear_regression.png)

# # Problem

# **Experience and salary**
# 
# > **Description:**
# *How much must be salary of an employee in a compny based on experience*

# # Dataset preview

# | YearsExperience | Salary    |
# |-----------------|-----------|
# | 1.1             | 39343.00  |
# | 1.3             | 46205.00  |
# | 1.5             | 37731.00  |
# | 2.0             | 43525.00  |
# | 2.2             | 39891.00  |
# | 2.9             | 56642.00  |
# | 3.0             | 60150.00  |
# | 3.2             | 54445.00  |
# | 3.2             | 64445.00  |
# | 3.7             | 57189.00  |
# | 3.9             | 63218.00  |
# | 4.0             | 55794.00  |
# | 4.0             | 56957.00  |
# | 4.1             | 57081.00  |

# # Implementation

# # Importing the libraries

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# # Importing the dataset

# In[ ]:


dataset = pd.read_csv('../input/Salary.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


# # Splitting the dataset into the Training set and Test set

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# # Fitting Simple Linear Regression to the Training set

# In[ ]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# # Predicting the Test set results

# In[ ]:


y_pred = regressor.predict(X_test)


# # Visualising the Training set results

# In[ ]:


plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary/Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# # Visualising the Test set results

# In[ ]:


plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary/Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

