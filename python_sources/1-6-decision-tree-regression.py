#!/usr/bin/env python
# coding: utf-8

# # 1.6 Decision Tree Regression

# For a better understanding of Decision Tree if u don't have any idea whatsoever, visit the links:
# 
# [Decision Tree Intuition 1](https://www.geeksforgeeks.org/decision-tree/)
# 
# [Decision Tree Intuition 2](https://www.kdnuggets.com/2020/01/decision-tree-algorithm-explained.html)
# 

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
# 
# It basically tells u about the preprocessing & Linear Regression which will help u in understanding this notebook better

# Decision Tree works better with multiple feature data i.e X should be multiple columns.
# 
# Here we are going to go with single feature as it will have an advantage to visualise the result in a 2 Dimensional plot.
# 
# suppose if we have 5 features then there will be total 6 items including the output to show on the graph and it's not possible to show the 6 dimensions hence we are sticking to a single feature dataset. for our understanding.

# ![decitree.png](attachment:decitree.png)

# **Decision Tree unlike other models doesn't use mathematic equations to predict the value/ Traget as it uses a splitting of nodes and make branches and sub-nodes.**
# 
# **Hence we dont have to scale the data.**

# ![Decision_Tree-2.png](attachment:Decision_Tree-2.png)

# ## Importing the libraries

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# We will be using the same dataset used in the SVR and other to train the model and then predict the salary of a person who is in between level 6 & 7.

# In[ ]:


dataset = pd.read_csv('../input/position-salaries/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


# ## Training the Decision Tree Regression model on the whole dataset

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)


# There is a little tuning we can do for the model to work better, but let's go with basics here.
# 
# We are fixing the random seed/ random_state to some value so that we can reproduce the same results everytime.
# 
# 

# ## Predicting a new result

# let's predict the salary of a person who is in between level 6 & 7

# In[ ]:


regressor.predict([[6.5]])


# ## Visualising the Decision Tree Regression results (higher resolution)

# In[ ]:


X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
'''
Here we are predicting each and every value of X starting from level 1 till level 10 with an interval of 0.01 to know the outcome and plot each value, as we plot each value we can
see that the model is predicting same outcome for level 1.5 to 2.5 as salary of a level 2, 2.5 to 3.5 as outcome of salary of level 3
If we haven't predicted outcome of each value we will get a graph which is drawn in the next 
code cell and there is very less to interpret from that.
'''
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# From the plot we can say that the model has trained in a way to predict and this is how it goes:
# 
# from level 3.5 to 4.5 it predicts every feature value as same as level 4 it goes on for all levels.
# 
# for level 6.5 it predicted the target as 150000 because the model thought that from level 5.5 to 6.5 every feature value will be equal to level 6 i.e the model thinks of a range which will be a node and if a certain feature in that node yes then certain output if not more splitting of nodes and branches.
# 
# Hence this decision tree is prefered for more features insted of a single feature.
# 
# 

# # Visualising the Decision Tree Regressor 

# Visualising in general which doesn't make any sense so better go with the higher resolution model. as it doesn't show the stepping range for each point.

# In[ ]:


plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# # Like this notebook then upvote it.
# 
# # Need to improve it then comment below.
# 
# # Enjoy Machine Learning
