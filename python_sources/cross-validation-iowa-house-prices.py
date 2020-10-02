#!/usr/bin/env python
# coding: utf-8

# *This tutorial is part Level 2 in the [Learn Machine Learning](https://www.kaggle.com/learn/machine-learning) curriculum. This tutorial picks up where Level 1 finished, so you will get the most out of it if you've done the exercise from Level 1.*
# 
# In this step, you will learn three approaches to dealing with missing values. You will then learn to compare the effectiveness of these approaches on any given dataset.* 
# 
# # Introduction
# 
# There are many ways data can end up with missing values. For example
# - A 2 bedroom house wouldn't include an answer for _How large is the third bedroom_
# - Someone being surveyed may choose not to share their income
# 
# Python libraries represent missing numbers as **nan** which is short for "not a number".  You can detect which cells have missing values, and then count how many there are in each column with the command:
# ```
# missing_val_count_by_column = (data.isnull().sum())
# print(missing_val_count_by_column[missing_val_count_by_column > 0
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
# from sklearn.impute import SimpleImputer
# my_imputer = SimpleImputer()
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
#                                  if new_data[col].isnull().any())
# for col in cols_with_missing:
#     new_data[col + '_was_missing'] = new_data[col].isnull()
# 
# # Imputation
# my_imputer = SimpleImputer()
# new_data = pd.DataFrame(my_imputer.fit_transform(new_data))
# new_data.columns = original_data.columns
# ```
# 
# In some cases this approach will meaningfully improve results. In other cases, it doesn't help at all.
# 
# ---
# # Example (Comparing All Solutions)
# 
# We will see am example predicting housing prices from the Melbourne Housing data.  To master missing value handling, fork this notebook and repeat the same steps with the Iowa Housing data.  Find information about both in the **Data** section of the header menu.
# 
# 
# ### Basic Problem Set-up
# Dataset :  **'Iowa House Prices'**.  
# 
# Model : **RandomForest using Pipeline and Cross-Validation**

# In[ ]:


import pandas as pd

# Load data
iowa_data = pd.read_csv('../input/iowa-house-prices/train.csv')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

iowa_target = iowa_data.SalePrice
iowa_predictors = iowa_data.drop(['SalePrice'], axis=1)

# For the sake of keeping the example simple, we'll use only numeric predictors. 
iowa_numeric_predictors = iowa_predictors.select_dtypes(exclude=['object'])


# ### Create Function to Measure Quality of An Approach
# We divide our data into **training** and **test**. If the reason for this is unfamiliar, review [Welcome to Data Science](https://www.kaggle.com/dansbecker/welcome-to-data-science-1).
# 
# We've loaded a function `score_dataset(X_train, X_test, y_train, y_test)` to compare the quality of diffrent approaches to missing values. This function reports the out-of-sample MAE score from a RandomForest.

# In[ ]:


from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

def score_dataset(X, y):
    my_pipeline = make_pipeline(SimpleImputer(),RandomForestRegressor())
    scores = cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error')
    return scores

print("IOWA-HOUSE-PRICES: Mean Absolute Error from Cross-Validation:")
print(score_dataset(iowa_numeric_predictors, iowa_target))
print("Mean absolute error %2f " % (-1 * score_dataset(iowa_numeric_predictors, iowa_target).mean()))


# ### Get Score from Imputation with Extra Columns Showing What Was Imputed

# In[ ]:


imputed_X_plus = iowa_numeric_predictors.copy()

cols_with_missing = (col for col in iowa_numeric_predictors.columns 
                                 if iowa_numeric_predictors[col].isnull().any())
for col in cols_with_missing:
    imputed_X_plus[col + '_was_missing'] = imputed_X_plus[col].isnull()

print("IOWA-HOUSE-PRICES: Mean Absolute Error from Imputation while Track What Was Imputed using PIPELINE:")
print(score_dataset(imputed_X_plus, iowa_target))
print("Mean absolute error %2f " % (-1 * score_dataset(imputed_X_plus, iowa_target).mean()))


# # Conclusion
# As is common, imputing missing values allowed us to improve our model compared to dropping those columns.  We got an additional boost by tracking what values had been imputed.

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
# 
# # Keep Going
# Once you've added the Imputer and included columns with missing values, you are ready to [add categorical variables](https://www.kaggle.com/dansbecker/using-categorical-data-with-one-hot-encoding), which is non-numeric data representing categories (like the name of the neighborhood a house is in).
# 
# ---
# 
# Part of the **[Learn Machine Learning](https://www.kaggle.com/learn/machine-learning)** track.
