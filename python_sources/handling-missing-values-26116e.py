#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

# Load data
melb_data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

melb_target = melb_data.Price
melb_predictors = melb_data.drop(['Price'], axis=1)

# For the sake of keeping the example simple, we'll use only numeric predictors. 
melb_numeric_predictors = melb_predictors.select_dtypes(exclude=['object'])


# ### Create Function to Measure Quality of An Approach
# We divide our data into **training** and **test**. If the reason for this is unfamiliar, review [Welcome to Data Science](https://www.kaggle.com/dansbecker/welcome-to-data-science-1).
# 
# We've loaded a function `score_dataset(X_train, X_test, y_train, y_test)` to compare the quality of diffrent approaches to missing values. This function reports the out-of-sample MAE score from a RandomForest.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(melb_numeric_predictors, 
                                                    melb_target,
                                                    train_size=0.7, 
                                                    test_size=0.3, 
                                                    random_state=0)

def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)


# # Your Turn
# 1) Find some columns with missing values in your dataset.
# 
# 2) Use the Imputer class so you can impute missing values
# 
# 3) Add columns with missing values to your predictors. 
# 
# If you find the right columns, you may see an improvement in model scores. That said, the Iowa data doesn't have a lot of columns with missing values.  So, whether you see an improvement at this point depends on some other details of your model.
# 
# Once you've added the Imputer, keep using those columns for future steps.  In the end, it will improve your model (and in most other datasets, it is a big improvement). 

# In[ ]:


#1) Find some columns with missing values in your dataset.
cols_with_missing = [col for col in X_train.columns 
                                 if X_train[col].isnull().any()]
cols_with_missing
#2) Use the Imputer class so you can impute missing values
from sklearn.preprocessing import Imputer
my_imputer = Imputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)
#3) Add columns with missing values to your predictors. 
print("Mean Absolute Error from Imputation:")
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))


# 
