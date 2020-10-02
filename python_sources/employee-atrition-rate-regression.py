#!/usr/bin/env python
# coding: utf-8

# ## Libraries

# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor 
from sklearn.impute import SimpleImputer


# ## Configuration Files

# In[ ]:


train_df = pd.read_csv('../input/Dataset/Train.csv')
test_df = pd.read_csv('../input/Dataset/Test.csv')

train_df.head()
train_df.describe()


# ## Train and Validation Data

# In[ ]:


# req_columns = ['Gender','Age', 'Education_Level', 'Relationship_status', 'Hometown', 'Unit', 'Decision_skill_posses', 
#                'time_of_service', 'Time_since_promotion']
req_cols = ['Age', 'Education_Level', 'Time_of_service', 'Time_since_promotion', 'growth_rate', 'Travel_Rate', 'Post_Level',
           'Pay_Scale', 'Work_Life_balance', 'VAR1', 'VAR2', 'VAR3', 'VAR4', 'VAR5','VAR6']

X = train_df[req_cols]
y = train_df.Attrition_rate

train_X, valid_X, train_y, valid_y = train_test_split(X, y, random_state=0)
print("Size of total  {}, Size of train data {}, Size of vaid data {}".format(len(X), len(train_X), len(valid_X)))


# ## Missing Value Analysis and Correction

# In[ ]:


missing_valid_cols = [col for col in valid_X.columns if valid_X[col].isnull().any()]
print("missing_valid_cols ",missing_valid_cols)
missing_train_cols = [col for col in train_X.columns if train_X[col].isnull().any()]
print("missing_valid_cols ",missing_train_cols)

## Imputing missing values
imputer = SimpleImputer()
new_train_X = pd.DataFrame(imputer.fit_transform(train_X))
new_valid_X = pd.DataFrame(imputer.transform(valid_X))
## Adding columns
new_train_X.columns = train_X.columns
new_valid_X.columns = valid_X.columns
train_X = new_train_X
valid_X = new_valid_X


# ## Baseline Model Using Decision Tree Regressor

# In[ ]:


def train_and_evaluate(train_X, train_y, valid_X, valid_y, max_leaf=500):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf, random_state=0)
    ## Fitting model
    model.fit(train_X, train_y)
    ## Training Loss
    train_preds = model.predict(train_X)
    train_loss = mean_absolute_error(train_preds, train_y)
    predictions = model.predict(valid_X)
    valid_loss = mean_absolute_error(predictions, valid_y)
    print("Base Line model performance \n \t \t----------------------------- \n \t                  Train loss {:,.4f} \t Valid loss {:,.4f}".format(train_loss, valid_loss))
    return model


# In[ ]:


for i in [10, 25, 50, 100, 150, 250, 500, 250, 1000]:
    print(i)
    train_and_evaluate(train_X, train_y, valid_X, valid_y, max_leaf=i)


# In[ ]:


## Best model
model = train_and_evaluate(train_X, train_y, valid_X, valid_y, max_leaf=10)


# ## Prediction

# In[ ]:


req_cols = ['Age', 'Education_Level', 'Time_of_service', 'Time_since_promotion', 'growth_rate', 'Travel_Rate', 'Post_Level',
           'Pay_Scale', 'Work_Life_balance', 'VAR1', 'VAR2', 'VAR3', 'VAR4', 'VAR5','VAR6']

test_X_plus = test_df[req_cols]

cols_missing = [col for col in test_X_plus.columns if test_X_plus[col].isnull().any()]
print(cols_missing)

## Imputing
new_test_X = pd.DataFrame(imputer.transform(test_X_plus))
## Adding columns
new_test_X.columns = test_X_plus.columns


# In[ ]:


def prediction(model, test_X):
    preds = model.predict(test_X)
    return preds
preds = prediction(model, new_test_X)
print(preds)


# In[ ]:


def submit(preds, test_df, filename = 'submission.csv'):
    ## Employee_ID   Attrition_rate
    empId = test_df['Employee_ID'].tolist()
    dict = {"Employee_ID": empId, "Attrition_rate": preds}
    sub = pd.DataFrame(dict)
    sub.to_csv(filename, index=False)
submit(preds, test_df)


# In[ ]:





# In[ ]:





# In[ ]:




