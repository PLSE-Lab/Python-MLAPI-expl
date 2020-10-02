#!/usr/bin/env python
# coding: utf-8

# [Housing Prices Competition for Kaggle Learn Users](https://www.kaggle.com/c/home-data-for-ml-course). 
# 
# ![Ames Housing dataset image](https://i.imgur.com/lTJVG4e.png)
# 
# 

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

# Read the data
X_train_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X_train_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_train_full.SalePrice
X_train_full.drop(['SalePrice'], axis=1, inplace=True)

# Grasp train and test
X_full = pd.concat([X_train_full,X_test_full])


# In[ ]:


# Shape of training data (num_rows, num_columns)
print(X_train_full.shape)

# Shape of test data (num_rows, num_columns)
print(X_test_full.shape)

# Shape of full data (num_rows, num_columns)
print(X_full.shape)

# Number of missing values in each column of full data
missing_val_count_by_column = (X_full.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])


# In[ ]:


cols_to_drop = ['Alley','PoolQC','Fence','MiscFeature']
X_full_mod = X_full.drop(cols_to_drop, axis=1)


# In[ ]:


# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X_full_mod.columns if
                    X_full_mod[cname].nunique() <= 20 and 
                    X_full_mod[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_full_mod.columns if 
                X_full_mod[cname].dtype in ['int64', 'float64']]


# In[ ]:


# replacing missing values with forward/backward fill method with pandas
X_full_fill = X_full_mod.fillna(method='ffill')
X_full_fill = X_full_fill.fillna(method='bfill')


# In[ ]:


# categorical dataset
cat = X_full_fill[categorical_cols]


# In[ ]:


# numerical dataset
num = X_full_fill[numerical_cols]


# In[ ]:


#polynomial features
poly_num2 = []
poly_num3 = []
poly_num4 = []
poly_num5 = []


for f in numerical_cols:
    poly_num2 = num[numerical_cols]**2
    poly_num3 = num[numerical_cols]**3
    poly_num4 = num[numerical_cols]**4
    poly_num5 = num[numerical_cols]**5
    poly_num2.columns = [col + '_2' for col in num.columns]
    poly_num3.columns = [col + '_3' for col in num.columns]
    poly_num4.columns = [col + '_4' for col in num.columns]
    poly_num5.columns = [col + '_5' for col in num.columns]
    poly_num2.append(numerical_cols)
    poly_num3.append(numerical_cols)
    poly_num4.append(numerical_cols)
    poly_num5.append(numerical_cols)


# In[ ]:


poly_num = pd.concat([num, poly_num2, poly_num3, poly_num4, poly_num5], axis=1, join='inner')


# In[ ]:


# One-hot encode the data (to shorten the code, use pandas)
HOcat = pd.get_dummies(cat)


# In[ ]:


# All dataset modified
X = pd.concat([HOcat,poly_num], axis=1, join='inner')


# In[ ]:


# train and test modified
X_train = X.loc[0:1460,:]
X_test = X.loc[1461:2919,:]


# In[ ]:


# Break off validation set from training data
X_train_, X_valid_, y_train, y_valid = train_test_split(X_train, y, 
                                                                train_size=0.8, test_size=0.2,
                                                                random_state=0)


# In[ ]:


# Define the model
def get_mae(n_estimators, X_train_, X_valid_, y_train, y_valid, learning_rate):
    xgb_model = XGBRegressor(n_estimators=n_estimators, learning_rate=0.02, cv=20, random_state=0)
    xgb_model.fit(X_train_, y_train)
    xgb_preds_val = xgb_model.predict(X_valid_)
    mae = mean_absolute_error(y_valid, xgb_preds_val)
    return(mae)

score = []
c_n_estimators = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 
                  1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000]

# Write loop 
for f in c_n_estimators:   
    my_mae = get_mae(f, X_train_, X_valid_, y_train, y_valid, learning_rate=0.02)
    # creating a dictionary that contains all the metadata for the variable
    f_dict = {
        'n_estimators': f,
        'mean_absolute_error': my_mae,
    }
    score.append(f_dict)

# collecting all information into a meta dataframe
score = pd.DataFrame(score, columns = ['n_estimators','mean_absolute_error'])

# Store the best value 
xgb_results = {n_estimators: get_mae(n_estimators, X_train_, X_valid_, y_train, y_valid, learning_rate=0.02) for n_estimators in c_n_estimators}
best_xgb_n_estimators = min(xgb_results, key=xgb_results.get)
print(score)
print(best_xgb_n_estimators)

my_xgb_model = XGBRegressor(n_estimators=best_xgb_n_estimators, learning_rate=0.02, cv=20, random_state=0)

# Fit the model
my_xgb_model.fit(X_train_, y_train) 

# Get predictions
xgb_predictions = my_xgb_model.predict(X_valid_) 

# Calculate MAE
mae = mean_absolute_error(y_valid, xgb_predictions) # Your code here

# Uncomment to print MAE
print("Mean Absolute Error:" , mae)


# # Step 2: Generate test predictions
# 
# Now, you'll use your trained model to generate predictions with the test data.

# In[ ]:


# Preprocessing of test data, fit model
preds_test = my_xgb_model.predict(X_test) 


# In[ ]:


# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)

