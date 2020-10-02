#!/usr/bin/env python
# coding: utf-8

# In this step, I will take three approaches to dealing with missing values. This is part of a task into learning how to produce efficient Machine Learning algorithms.
# 
# Further, I will add one-hot-encoding to my model. This is basically predict the target using categorical variables in my predictors features set. 
# 
# # Handling missing values
# 
# There are many ways data can end up with missing values. For example
# - A 2 bedroom house wouldn't include an answer for _How large is the third bedroom_
# - Someone being surveyed may choose not to share their income
# 
# Python libraries represent missing numbers as **nan** which is short for "not a number".  You can detect which cells have missing values, and then count how many there are in each column with the command:
# ```
# print(data.isnull().sum())
# ```
# 
# Most libraries (including scikit-learn) will give you an error if you try to build a model using data with missing values. So you'll need to choose one of the strategies below.
# 
# ---
# ## Solutions
# 
# 
# ## 1) A Simple Option: Drop Columns with Missing Values
# If your data is in a DataFrame called `original_data`, you can drop columns with missing values. One way to do that is
# ```
# data_without_missing_values = original_data.dropna(axis=1)
# ```
# 
# In many cases, you'll have both a training dataset and a test dataset.  You will want to drop the same columns in both DataFrames. In that case, you would write
# 
# ```
# cols_with_missing = [col for col in original_data.columns 
#                                  if original_data[col].isnull().any()]
# redued_original_data = original_data.drop(cols_with_missing, axis=1)
# reduced_test_data = test_data.drop(cols_with_missing, axis=1)
# ```
# If those columns had useful information (in the places that were not missing), your model loses access to this information when the column is dropped. Also, if your test data has missing values in places where your training data did not, this will result in an error.  
# 
# So, it's somewhat usually not the best solution. However, it can be useful when most values in a column are missing.
# 
# 
# 
# ## 2) A Better Option: Imputation
# Imputation fills in the missing value with some number. The imputed value won't be exactly right in most cases, but it usually gives more accurate models than dropping the column entirely.
# 
# This is done with
# ```
# from sklearn.preprocessing import Imputer
# my_imputer = Imputer()
# data_with_imputed_values = my_imputer.fit_transform(original_data)
# ```
# The default behavior fills in the mean value for imputation.  Statisticians have researched more complex strategies, but those complex strategies typically give no benefit once you plug the results into sophisticated machine learning models.
# 
# One (of many) nice things about Imputation is that it can be included in a scikit-learn Pipeline. Pipelines simplify model building, model validation and model deployment.
# 
# ## 3) An Extension To Imputation
# Imputation is the standard approach, and it usually works well.  However, imputed values may by systematically above or below their actual values (which weren't collected in the dataset). Or rows with missing values may be unique in some other way. In that case, your model would make better predictions by considering which values were originally missing.  Here's how it might look:
# ```
# # make copy to avoid changing original data (when Imputing)
# new_data = original_data.copy()
# 
# # make new columns indicating what will be imputed
# cols_with_missing = (col for col in new_data.columns 
#                                  if new_data[c].isnull().any())
# for col in cols_with_missing:
#     new_data[col + '_was_missing'] = new_data[col].isnull()
# 
# # Imputation
# my_imputer = Imputer()
# new_data = my_imputer.fit_transform(new_data)
# ```
# 
# In some cases this approach will meaningfully improve results. In other cases, it doesn't help at all.
# 
# ---
# # Comparing All Solutions
# 
# To master missing value handling, I will compare the three solutions applying them to the Iowa Housing data. 

# In[ ]:


import pandas as pd

# Load data
iowa_file_path = '../input/house-prices-advanced-regression-techniques/train.csv'
iowa_data = pd.read_csv(iowa_file_path) 

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

##Separating data in feature predictors, and target to predict
iowa_target = iowa_data.SalePrice
iowa_predictors = iowa_data.drop(['SalePrice'], axis=1)

# For the sake of keeping the example simple, we'll use only numeric predictors. 
iowa_numeric_predictors = iowa_predictors.select_dtypes(exclude=['object'])


# ### Create Function to Measure Quality of An Approach
# We divide our data into **training** and **test**. 
# 
# We've loaded a function `score_dataset(X_train, X_test, y_train, y_test)` to compare the quality of diffrent approaches to missing values. This function reports the out-of-sample MAE score from a RandomForest.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

### Separate dataset samples into train and test data 
X_train, X_test, y_train, y_test = train_test_split(iowa_numeric_predictors, 
                                                    iowa_target,
                                                    train_size=0.7, 
                                                    test_size=0.3, 
                                                    random_state=0)

### Function to score or evaluate how good its the model
def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)


# ### Get Model Score from Dropping Columns with Missing Values

# In[ ]:


cols_with_missing = [col for col in X_train.columns 
                                 if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_test  = X_test.drop(cols_with_missing, axis=1)
print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))


# ### Get Model Score from Imputation

# In[ ]:


from sklearn.preprocessing import Imputer

my_imputer = Imputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)
print("Mean Absolute Error from Imputation:")
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))


# ### Get Score from Imputation with Extra Columns Showing What Was Imputed

# In[ ]:


imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()

cols_with_missing = (col for col in X_train.columns 
                                 if X_train[col].isnull().any())
for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()

# Imputation
my_imputer = Imputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

print("Mean Absolute Error from Imputation while Track What Was Imputed:")
print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))


# # Conclusion
# In this case, the extension didn't make a big difference. As mentioned before, this can vary widely from one dataset to the next (largely determined by whether rows with missing values are intrinsically like or unlike those without missing values).

# # Adding One Hot Encoding to our model

# In[ ]:


##Separating data in feature predictors, and target to predict
iowa_target = iowa_data.SalePrice
iowa_predictors = iowa_data.drop(['SalePrice'], axis=1)

### Separate dataset samples into train and test data 
X_train2, X_test2, y_train2, y_test2 = train_test_split(iowa_predictors, 
                                                    iowa_target,
                                                    train_size=0.7, 
                                                    test_size=0.3, 
                                                    random_state=0)

### Function to score or evaluate how good its the model
def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)

### Getting train & test datasets for model evaluations: One-Hot-Encoding & dropping categorical features.
X_train_OHE = pd.get_dummies(X_train2)
X_test_OHE = pd.get_dummies(X_test2)


X_train_without_categoricals = X_train2.select_dtypes(exclude=['object'])
X_test_without_categoricals = X_test2.select_dtypes(exclude=['object'])

### Getting aligned train & test datasets for both model evaluations
final_train, final_test = X_train_OHE.align(X_test_OHE,
                                            join='left', 
                                            axis=1)
final_train_without_categorical, final_test_without_categorical = X_train_without_categoricals.align(X_test_without_categoricals,
                                                                    join='left', 
                                                                    axis=1)

### Imputing train & test datasets, filling NaN values for both model evaluations
imputed_final_train = my_imputer.fit_transform(final_train)
imputed_final_test = my_imputer.transform(final_test)

imputed_final_train_without_categorical = my_imputer.fit_transform(final_train_without_categorical)
imputed_final_test_without_categorical = my_imputer.transform(final_test_without_categorical)

### Printing Errors for One-Hot-Encoded datasets vs Without categorical features.
print("Mean Absolute Error from Imputation&OHE:")
print(score_dataset(imputed_final_train, imputed_final_test, y_train2, y_test2))

print("Mean Absolute Error from Imputation&WithoutCategorical:")
print(score_dataset(imputed_final_train_without_categorical, imputed_final_test_without_categorical, y_train2, y_test2))


# # Conclusion
# In this case, the one-hot-encoding didn't make a big difference. But improved slightly our predictor model. 

# In[ ]:




