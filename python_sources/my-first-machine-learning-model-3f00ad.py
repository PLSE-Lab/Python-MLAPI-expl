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
#data.describe()
data.head()


# In[ ]:


y=data.SalePrice
Lowa_Features  = ['LotArea','OverallQual','OverallCond','YearBuilt','TotalBsmtSF']
X = data[Lowa_Features]
X.describe()
X.head()


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
#from sklearn.tree import DecisionTreeRegressor
Lowa_Model = DecisionTreeRegressor(random_state=1)
Lowa_Model.fit(X,y)


# In[ ]:


print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(Lowa_Model.predict(X.head()))

