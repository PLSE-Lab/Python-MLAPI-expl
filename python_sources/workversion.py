#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
data_path = '../input/train.csv'
iowa_data = pd.read_csv(data_path) 
filtered_iowa_data = iowa_data


# In[31]:


y = filtered_iowa_data.SalePrice
iowa_predictors = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X = filtered_iowa_data[iowa_predictors]


# In[32]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
iowa_model = DecisionTreeRegressor()

iowa_model.fit(train_X, train_y)


# In[33]:


from sklearn.metrics import mean_absolute_error
print("The predictions are")
predictions=iowa_model.predict(val_X)
mean_absolute_error(val_y, predictions)

