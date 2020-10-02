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
print('hello world')


# In[ ]:





# In[ ]:


# save filepath to variable for easier access
Iowa_file_path = '../input/train.csv'
# read the data and store data in DataFrame titled Iowa_data
Iowa_data = pd.read_csv(Iowa_file_path) 
# print a summary of the data in Iowa data
#print(Iowa_data.describe())


# In[ ]:


Iowa_data.head()


# **Print a list of column**

# In[ ]:


print(Iowa_data.columns)


# From the list of columns, find a name of the column with the sales prices of the homes. Use the dot notation to extract this to a variable (as you saw above to create melbourne_price_data.)
# **Use the head command to print out the top few lines of the variable you just created.
# 

# In[ ]:





# In[ ]:


# store the series of prices separately as Iowa_price_data.
Iowa_price_data = Iowa_data.SalePrice
# the head command returns the top few lines of data.
print(Iowa_price_data.head())


# **Pick any two variables and store them to a new DataFrame (as you saw above to create two_columns_of_data.)
# **

# In[ ]:


columns_of_interest = ['GarageYrBlt', 'SalePrice']
two_columns_of_data = Iowa_data[columns_of_interest]


# In[ ]:


two_columns_of_data.describe()


# In[ ]:


# dropna drops missing values (think of na as "not available")
Iowa_data = Iowa_data.dropna(axis=0)


# In[ ]:


y = Iowa_data.SalePrice


# In[ ]:


Iowa_data.head


# In[ ]:


Iowa_features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']


# In[ ]:


X = Iowa_data[Iowa_features]


# In[ ]:


X.describe()


# In[ ]:


X.head()


# In[ ]:


from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
Iowa_model = DecisionTreeRegressor(random_state=1)


# Fit model
Iowa_model.fit(X, y)


# In[ ]:


print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(Iowa_model.predict(X.head()))


# In[ ]:




