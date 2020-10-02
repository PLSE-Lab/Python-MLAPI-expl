#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Code you have previously used to load data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from learntools.core import *

# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = '../input/train.csv'
home_data = pd.read_csv(iowa_file_path)

# path to file you will use for predictions
test_data_path = '../input/test.csv'
test_data = pd.read_csv(test_data_path)

y = home_data.SalePrice
X = home_data.iloc[:,:-1]


# In[ ]:


home_data.head(5)


# In[ ]:


X.head(5)


# In[ ]:


# Low cardinality columns
low_cardinality_cols = [col for col in home_data.columns
    if home_data[col].dtype == 'object' and home_data[col].nunique() < 10]
print(low_cardinality_cols)


# In[ ]:


ohe_train = pd.get_dummies(X)


# In[ ]:


from sklearn.impute import SimpleImputer
def impute_df(X):
    my_imputer = SimpleImputer()
    imputed_X = pd.DataFrame(my_imputer.fit_transform(X))
    imputed_X.columns = X.columns
    return imputed_X


# In[ ]:


# Impute data
imputed_ohe_train = impute_df(ohe_train)


# # Testing model

# In[ ]:


# Train-test split for validaiton
X_train, X_test, y_train, y_test = train_test_split(imputed_ohe_train, y, test_size=1/5, random_state=0)


# In[ ]:


# Helper function to get MAE
def get_mae(model, X_train, X_test, y_train, y_test):
    model = model
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    return(mae)


# ## One hot encoded data with RF

# In[ ]:


rf = RandomForestRegressor(max_leaf_nodes=50, n_estimators=100)
rf_mae = get_mae(rf, X_train, X_test, y_train, y_test)
print("MAE one hot encoded: {:.0f}".format(rf_mae))


# ## One hot encoded data with XGB

# In[ ]:


# Test xgb with different n_estimators
xgb = XGBRegressor(n_estimators=250, learning_rate=0.05)
xgb_mae = get_mae(xgb,X_train, X_test, y_train, y_test)
xgb.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_test, y_test)], verbose=False)
preds = xgb.predict(X_test)
xgb_mae = mean_absolute_error(y_test, preds)
print("XGB MAE: {:.0f}".format(xgb_mae))


# # Prepare test data

# In[ ]:


# Apply OHE to test data
test_data_with_features = test_data
ohe_test = pd.get_dummies(test_data_with_features)

# Impute nan values in test data 
imputed_ohe_test = impute_df(ohe_test)

final_train, final_test = imputed_ohe_train.align(imputed_ohe_test, join='left', axis=1)

#drop nan columns returned by aligning
cols_with_missing1 = final_test.columns[final_test.isna().any()].tolist()
final_test.drop(cols_with_missing1, axis=1, inplace=True)
final_train.drop(cols_with_missing1, axis=1, inplace=True)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(final_train, y, test_size=1/5, random_state=0)


# # Creating a Model For the Competition
# 
# Build a Random Forest model and train it on all of **X** and **y**.  

# In[ ]:


# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = RandomForestRegressor(50, random_state=1)

# fit rf_model_on_full_data on all data from the training data
rf_model_on_full_data.fit(imputed_ohe_train, y)

rf_model_on_full_data = XGBRegressor(n_estimators=250, learning_rate=0.05)
rf_model_on_full_data.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_test, y_test)], verbose=False)


# # Make Predictions
# Read the file of "test" data. And apply your model to make predictions

# In[ ]:


# make predictions which we will submit. 
test_preds = rf_model_on_full_data.predict(final_test)

# The lines below shows how to save predictions in format used for competition scoring
# Just uncomment them.

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)

