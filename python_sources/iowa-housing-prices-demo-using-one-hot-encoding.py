#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
# Loading in Iowa housing data
main_file_path = '../input/house-prices-advanced-regression-techniques/train.csv' 
# this is the path to the Iowa data that you will use
iowa_data = pd.read_csv(main_file_path)
print('Setup Complete...')


# In[ ]:


# import what we need for scikit to set up our model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

iowa_target = iowa_data.SalePrice
iowa_predictors = iowa_data.drop(['SalePrice'], axis=1)

# we will only use numeric predictors for this model
iowa_numeric_predictors = iowa_predictors.select_dtypes(exclude=['object'])
print(iowa_numeric_predictors.columns)


# First we will need to split our data into training and testing sets, then we create a function to compare the quality of the different approaches we will take to get rid of our missing values.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(iowa_numeric_predictors, 
                                                    iowa_target,
                                                    train_size=0.7, 
                                                    test_size=0.3, 
                                                    random_state=0)

def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)


# First we will train a model that simply drops any columns with missing values and get its MAE

# In[ ]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #so pandas doesn't spit out a warning everytime
cols_with_missing = [col for col in X_train.columns 
                                 if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_test  = X_test.drop(cols_with_missing, axis=1)
print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))


# Now we will get the MAE for our model that uses Imputation instead

# In[ ]:


from sklearn.preprocessing import Imputer as Imputer
my_imputer = Imputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)
print("Mean Absolute Error from Imputation:")
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))


# Now for fun we will get the model score for when we use imputation with extra columns showing

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


# # One-Hot Encoding
# **This MAE is better than our previous results, but we are still throwing out a bunch of data by only predicting our Sale price based on only our numeric data. To solve this and get more accurate results we will start over and implement One-Hot Encoding.**

# In[ ]:


# Read the data
# import pandas as pd #ALREADY IMPORTED ABOVE

train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

# Drop houses where the target is missing
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
target = train_data.SalePrice

# Since missing values isn't the focus of this tutorial, we use the simplest
# possible approach, which drops these columns. 
# For more detail (and a better approach) to missing values, see
# https://www.kaggle.com/dansbecker/handling-missing-values
cols_with_missing = [col for col in train_data.columns 
                                 if train_data[col].isnull().any()]                                  
candidate_train_predictors = train_data.drop(['Id', 'SalePrice'] + cols_with_missing, axis=1)
candidate_test_predictors = test_data.drop(['Id'] + cols_with_missing, axis=1)


# **"Cardinality" means the number of unique values in a column.
# We use it as our only way to select categorical columns here. This is convenient, though a little arbitrary.**

# In[ ]:


low_cardinality_cols = [cname for cname in candidate_train_predictors.columns if
                       candidate_train_predictors[cname].nunique() < 10 and
                       candidate_train_predictors[cname].dtype == 'object']
numeric_cols = [cname for cname in candidate_train_predictors.columns if 
                                candidate_train_predictors[cname].dtype in ['int64', 'float64']]
my_cols = low_cardinality_cols + numeric_cols
train_predictors = candidate_train_predictors[my_cols]
test_predictors = candidate_train_predictors[my_cols]


# Pandas assigns a data type (called a dtype) to each column or Series. Let's see a random sample of dtypes from our prediction data:

# In[ ]:


train_predictors.dtypes.sample(10)


# **Object** indicates a column has text (there are other things it could be theoretically be, but that's unimportant for our purposes). It's most common to one-hot encode these "object" columns, since they can't be plugged directly into most models. Pandas offers a convenient function called **get_dummies** to get one-hot encodings. Call it like this:

# In[ ]:


one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)


# Alternatively, we can drop these object columns and compare the MAE of the two methods to see which works best for our dataset:
# 
# 1. One-hot encoded categoricals as well as numeric predictors
# 
# 2. Numerical predictors, where we drop categoricals.
# 
# 

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

def get_mae(X,y):
    # we multiply by -1 in this instance in order to ouput a positve MAE score 
    # instead of a negative value returned by sklearn
    return -1 * cross_val_score(RandomForestRegressor(50),
                                X,y,
                                scoring = 'neg_mean_absolute_error').mean()

predictors_without_categoricals = train_predictors.select_dtypes(exclude=['object'])

mae_without_categoricals = get_mae(predictors_without_categoricals, target)

mae_one_hot_encoded = get_mae(one_hot_encoded_training_predictors, target)

print('Mean Absolute Error when Dropping Categoricals: ' + str(int(mae_without_categoricals)))
print('Mean Absolute Error with One-Hot Encoding: ' + str(int(mae_one_hot_encoded)))

# print('MSE: ' + mean_squared_error(one_hot_encoded_training_predictors, target))


# # Applying to Multiple Files
# So far, you've one-hot-encoded your training data. What about when you have multiple files (e.g. a test dataset, or some other data that you'd like to make predictions for)? Scikit-learn is sensitive to the ordering of columns, so if the training dataset and test datasets get misaligned, your results will be nonsense. This could happen if a categorical had a different number of values in the training data vs the test data.
# 
# **Ensure the test data is encoded in the same manner as the training data with the align command:**

# In[ ]:


one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                   
                                                                   join='left',
                                                                   
                                                                   axis=1)


# The align command makes sure the columns show up in the same order in both datasets (it uses column names to identify which columns line up in each dataset.) The argument join='left' specifies that we will do the equivalent of SQL's left join. That means, if there are ever columns that show up in one dataset and not the other, we will keep exactly the columns from our training data. The argument join='inner' would do what SQL databases call an inner join, keeping only the columns showing up in both datasets. That's also a sensible choice.

# 

# 

# 
