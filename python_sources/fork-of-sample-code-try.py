#!/usr/bin/env python
# coding: utf-8

# Sample of code below.

# In[ ]:


import pandas as pd
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
print(melbourne_data.describe())
print(melbourne_data.columns)
melbourne_price_data = melbourne_data.Price
print(melbourne_price_data.head())
columns_of_interest=['Landsize', 'BuildingArea','Rooms']
two_columns_of_data=melbourne_data[columns_of_interest]
two_columns_of_data


# In[ ]:


melbourne_predictors = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']
x=melbourne_data[melbourne_predictors]
print(x)
x.isnull().sum()
#y=melbourne_data.Price
#from sklearn.tree import DecisionTreeRegressor
#melbourne_model = DecisionTreeRegressor()
#melbourne_model.fit(x, y)


# In[ ]:


x['Bathroom'].fillna(x['Bathroom'].mean(), inplace=True)
x['Landsize'].fillna(x['Landsize'].mean(), inplace=True)
x['BuildingArea'].fillna(x['BuildingArea'].mean(), inplace=True)
x['YearBuilt'].fillna(x['YearBuilt'].mean(), inplace=True)
x['Lattitude'].fillna(x['Lattitude'].mean(), inplace=True)
x['Longtitude'].fillna(x['Longtitude'].mean(), inplace=True)
x.isnull().sum()


# In[ ]:


y=melbourne_data.Price
from sklearn.tree import DecisionTreeRegressor
melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(x, y)


# In[ ]:


print("Making predictions for the following 5 houses:")
print(x.head())
print("The predictions are")
print(melbourne_model.predict(x.head()))


# In[ ]:





# In[ ]:


from sklearn.metrics import mean_absolute_error
predicted_home_prices = melbourne_model.predict(x)
mean_absolute_error(y, predicted_home_prices)


# In[ ]:


from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(x, y,random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)
# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


# In[ ]:


from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)


# In[ ]:


for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))


# In[ ]:




