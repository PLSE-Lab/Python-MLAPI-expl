#!/usr/bin/env python
# coding: utf-8

# Fork of Kaggle/Learn/ML Advanced Regression
# 
# This is a quick and dirty baseline that is missing many basic pieces. It doesn't even use categorical variables (year, department number) correctly.

# In[ ]:


import pandas as pd

data = pd.read_table('../input/TrainingDataSet_Maize.csv', index_col=0)


# In[ ]:


data.describe()


# ### Select some columns of interest

# In[ ]:


y = data.yield_anomaly
columns_of_interest = ['year_harvest', 'NUMD', 'IRR', 'ETP_1', 'ETP_2',
       'ETP_3', 'ETP_4', 'ETP_5', 'ETP_6', 'ETP_7', 'ETP_8', 'ETP_9', 'PR_1',
       'PR_2', 'PR_3', 'PR_4', 'PR_5', 'PR_6', 'PR_7', 'PR_8', 'PR_9', 'RV_1',
       'RV_2', 'RV_3', 'RV_4', 'RV_5', 'RV_6', 'RV_7', 'RV_8', 'RV_9',
       'SeqPR_1', 'SeqPR_2', 'SeqPR_3', 'SeqPR_4', 'SeqPR_5', 'SeqPR_6',
       'SeqPR_7', 'SeqPR_8', 'SeqPR_9', 'Tn_1', 'Tn_2', 'Tn_3', 'Tn_4', 'Tn_5',
       'Tn_6', 'Tn_7', 'Tn_8', 'Tn_9', 'Tx_1', 'Tx_2', 'Tx_3', 'Tx_4', 'Tx_5',
       'Tx_6', 'Tx_7', 'Tx_8', 'Tx_9']
X = data[columns_of_interest]
X.describe()


# In[ ]:


y.describe()


# ### Build a simple Decision Tree model

# In[ ]:


from sklearn.tree import DecisionTreeRegressor

# Define model
dtr_model = DecisionTreeRegressor()

# Fit model
dtr_model.fit(X, y)


# In[ ]:


print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(dtr_model.predict(X.head()))


# ### Model validation using MAE

# In[ ]:


from sklearn.metrics import mean_absolute_error

predicted_yield_anoms = dtr_model.predict(X)
# In-sample error
mean_absolute_error(y, predicted_yield_anoms)


# ### Let's setup a validation dataset and compute MAE on it

# In[ ]:


from sklearn.model_selection import train_test_split

# split data into training, test and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state = 0, test_size=500)
train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, random_state = 0, test_size=500)


print(len(X), len(y))
print(len(train_X), len(train_y))
print(len(test_X), len(test_y))
print(len(val_X), len(val_y))
# Define model
dtr_model = DecisionTreeRegressor()
# Fit model
dtr_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = dtr_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


# ## Experimenting With Different Models
# 
# We can test different models and tune hyperparameters. An utility function will help.

# In[ ]:


from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return mae


# #### Now we can tune max_leaf_nodes

# In[ ]:


# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [2, 2, 3, 5, 50, 100, 500]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: {}  \t\t Mean Absolute Error:  {}".format(max_leaf_nodes, my_mae))


# In[ ]:


# Let's search some more around 50
for max_leaf_nodes in [5, 10, 20, 50, 100, 200]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: {}  \t\t Mean Absolute Error:  {}".format(max_leaf_nodes, my_mae))


# In[ ]:


# Let's search some more around 50
for max_leaf_nodes in [20, 30, 40, 45, 50, 55, 60, 70]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: {}  \t\t Mean Absolute Error:  {}".format(max_leaf_nodes, my_mae))


# #### So we'll pick max_leaf_nodes = 20
# This value is tuned for this model on this dataset.
# 
# ## Random Forest
# This is a much better model for this regression problem

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
yield_anom_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, yield_anom_preds))


# In[ ]:


def get_mae_rf(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return mae


# ### Submit to the competition

# In[ ]:


# Read the test data
test = pd.read_table('../input/TestDataSet_Maize_blind.csv')
# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test[columns_of_interest]
# Use the model to make predictions
predicted_anoms = forest_model.predict(test_X)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_anoms)


# In[ ]:


test.describe()


# In[ ]:


#my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

#my_submission.to_csv('submission.csv', index=False)

