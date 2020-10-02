#!/usr/bin/env python
# coding: utf-8

# # Introduction
# **This will be your workspace for Kaggle's Machine Learning education track.**
# 
# You will build and continually improve a model to predict housing prices as you work through each tutorial.  Fork this notebook and write your code in it.
# 
# The data from the tutorial, the Melbourne data, is not available in this workspace.  You will need to translate the concepts to work with the data in this notebook, the Iowa data.
# 
# Come to the [Learn Discussion](https://www.kaggle.com/learn-forum) forum for any questions or comments. 
# 
# # Write Your Code Below
# 
# 

# In[ ]:


import pandas as pd

main_file_path = '../input/train.csv'
data = pd.read_csv(main_file_path)
print(data.describe())


# In[ ]:


data.columns


# In[ ]:


data.head(1)


# In[ ]:


y = data.SalePrice
y.head(5)


# In[ ]:


data_features = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt']
X = data[data_features]
X.describe()


# In[ ]:


X.head(5)


# In[ ]:


from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
data_model = DecisionTreeRegressor(random_state=1)

# Fit model
data_model.fit(X, y)


# In[ ]:


print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(data_model.predict(X.head()))

