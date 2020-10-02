#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
heart_disease = pd.read_csv("../input/heart-disease-uci/heart.csv")


# In[ ]:


heart_disease.head()


# In[ ]:


heart_disease.columns


# In[ ]:


y = heart_disease.chol


# In[ ]:


heart_disease = heart_disease.dropna(axis=0)


# In[ ]:


heart_features = ['chol']


# In[ ]:


X = heart_disease [heart_features]


# In[ ]:


X.describe()


# In[ ]:


X.head()


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
heart_model = DecisionTreeRegressor(random_state=1)
heart_model.fit(X, y)


# In[ ]:


print("The chol 5 following numbers :")
print(X.head())
print("Numbers are")
print(heart_model.predict(X.head()))


# In[ ]:


from sklearn.metrics import mean_absolute_error

predicted_home_prices = heart_model.predict(X)
mean_absolute_error(y, predicted_home_prices)


# In[ ]:


from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
melbourne_model = DecisionTreeRegressor()
# Fit model
married_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = heart_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


# In[ ]:


from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


# In[ ]:


# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))

