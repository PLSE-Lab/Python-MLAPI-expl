#!/usr/bin/env python
# coding: utf-8

# # 1.7 Random Forest Regression

# For a better understanding of Random Forest if u don't have any idea whatsoever, visit the links:
# [Random Forest Intuition](https://www.youtube.com/watch?v=LIPtRVDmj1M)

# we can say that random forset is a whole large group of decision trees built from the subset of the dataset and their target is predicted by taking the majority of decision tree target in case of clasiificaion and average of decision tree target incase of regression.

# For better understanding of current notebook for beginners go through the links:
# 
#  [1.1 Data Preprocessing](http://www.kaggle.com/saikrishna20/data-preprocessing-tools)
# 
# 
# [1.2 Simple linear Regression](https://www.kaggle.com/saikrishna20/1-2-simple-linear-regression) 
# 
# 
# [1.3 Multiple linear Regression with Backward Elimination](http://www.kaggle.com/saikrishna20/1-3-multiple-linear-regression-backward-eliminat)
# 
# [1.4 Polynomial Linear Regression](https://www.kaggle.com/saikrishna20/1-4-polynomial-linear-regression)
# 
# [1.5 Support Vector Regression (SVR)](https://www.kaggle.com/saikrishna20/1-5-support-vector-regression-svr/edit/run/37240657)
# 
# [1.6 Decision Tree Regressor](https://www.kaggle.com/saikrishna20/1-6-decision-tree-regression)
# Definetely go through the decision tree link
# 
# It basically tells u about the preprocessing & Linear Regression which will help u in understanding this notebook better

# ## Importing the libraries

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[ ]:


dataset = pd.read_csv('../input/position-salaries/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


# ## Training the Random Forest Regression model on the whole dataset

# n_estimators (int) default=100
# The number of trees(Decision Trees) in the forest.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)


# ## Predicting a new result

# In[ ]:


regressor.predict([[6.5]])


# ## Visualising the Random Forest Regression results (higher resolution)

# Here we will predict each value i.e. 1.01 , 1.02, 1.03,1.04 and so.... on untill 10.00 and plot these outcomes for a wider understanding.

# In[ ]:


X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# As the Position level is increasing there are more steps in the model.
# 
# 1.5 to 2.5 salary is same
# 
# 2.5 to 3.0 
# 
# 3.0 to 3.5
# 
# 3.5 to 4.0
# 
# 4.0 to 4.5
# 
# 4.5 to 5.0 
# 
# 5.0 to 5.5 
# 
# u can clearly see the change in 7 to 7.5 and 7.5 to 8.0 there is a hike in salary.
# 
# In the decision tree the range is constant and is higher.
# 
# In the random forest the range is constant and is lower for a certain values whose salary is same, that's what the model is pedicting by using a 10 decision trees. By combining multiple trees we get forest.

# # Like this notebook then upvote it.
# 
# 
# # Need to improve it then comment below.
# 
# 
# # Enjoy Machine Learning
