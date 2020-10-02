#!/usr/bin/env python
# coding: utf-8

# # Learning Machine Learning
# **Workspace for the [Machine Learning course](https://www.kaggle.com/learn/machine-learning).**

# Read data into pandas DataFrames.

# In[ ]:


import pandas as pd

main_file_path = '../input/house-prices-advanced-regression-techniques/train.csv' # this is the path to the Iowa data that you will use
data = pd.read_csv(main_file_path)

# Run this code block with the control-enter keys on your keyboard. Or click the blue botton on the left
print('Some output from running this cell')
print(data.columns)

data_price = data.SalePrice
print(data_price.head())


# In[ ]:


column_names = ['LotArea', 'SalePrice']
condition_and_price = data[column_names]

condition_and_price.describe()


# We come up with a list of important features. How to best determine those features is a difficult problem in itself.

# In[ ]:


# data = data.dropna(axis=0)
y = data.SalePrice

estimators = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
X = data[estimators]


# In[ ]:


from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()
model.fit(X, y)

print("making predictions")
print(X.head())
print("the predictions are")
print(model.predict(X.head()))


# In[ ]:


from sklearn.metrics import mean_absolute_error

predicted_home_prices = model.predict(X)
mean_absolute_error(y, predicted_home_prices)


# We don't want to evaluate error on training data.
# 
# Scikit learn has built in way to split data into training data and validation data.

# In[ ]:


from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

model = DecisionTreeRegressor()
model.fit(train_X, train_y)

val_predictions = model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


# Add two helper functions to determine best value for parameter: max_leaf_nodes.

# In[ ]:


from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return mae

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d \t\t Mean Absolute Error: %f" %(max_leaf_nodes, my_mae))


# Using **RandomForest** model to see some performance improvements.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
forest_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, forest_preds))


# Without any tuning, RandomForest gives better results (24322) than best results of decision tree (27825).

# ## Submitting a model
# 
# First train the model on training data.

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

train_y = train.SalePrice

predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']

train_X = train[predictor_cols]

my_model = RandomForestRegressor()
my_model.fit(train_X, train_y)


# Then evaluate model on test data.

# In[ ]:


test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

test_X = test[predictor_cols]

predicted_prices = my_model.predict(test_X)

print(predicted_prices)


# In[ ]:


my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

my_submission.to_csv('submission.csv', index=False)


# ## Handling missing values
# 
# ### Problem setup

# In[ ]:


import pandas as pd

data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

target = data.SalePrice
predictors = data.drop(['SalePrice'], axis=1)

# only consider numeric columns
numeric_predictors = predictors.select_dtypes(exclude=['object'])


# ### Helper functions to measure quality

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(numeric_predictors, 
                                                    target,
                                                    train_size=0.7,
                                                    test_size=0.3,
                                                    random_state=0)

def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)


# ### Dropping columns with missing values

# In[ ]:


cols_with_missing = [col for col in X_train.columns
                                 if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_test = X_test.drop(cols_with_missing, axis=1)
print("Mean absolute error from dropping columsn with missing values:")
print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))


# ### Imputation

# In[ ]:


from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)
print("Mean absolute error from imputation:")
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))


# ### Imputation with extra columns showing what was imputed

# In[ ]:


imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()

cols_with_missing = (col for col in X_train.columns
                                 if X_train[col].isnull().any())

for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()
    
my_imputer = SimpleImputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

print("Mean absolute error from imputation while track imputed cols:")
print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))


# ### Handling categorical data

# In[ ]:


import pandas as pd
train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

# Drop data samples where SalePrice is empty.
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
target = train_data.SalePrice

cols_with_missing = [col for col in train_data.columns if train_data[col].isnull().any()]
cand_train_predictors = train_data.drop(['Id', 'SalePrice'] + cols_with_missing, axis=1)
cand_test_predictors = test_data.drop(['Id'] + cols_with_missing, axis=1)

low_card_cols = [cname for cname in cand_train_predictors.columns if 
                     cand_train_predictors[cname].nunique() < 10 and
                     cand_train_predictors[cname].dtype == "object"]
numeric_cols = [cname for cname in cand_train_predictors.columns if
                   cand_train_predictors[cname].dtype in ['int64', 'float64']]
my_cols = low_card_cols + numeric_cols
train_predictors = cand_train_predictors[my_cols]
test_predictors = cand_test_predictors[my_cols]


# In[ ]:


train_predictors.dtypes.sample(10)


# In[ ]:


one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)


# Comparing: 
# 1. one hot encoded categoricals + numeric predictors
# 2. numerical predictors only
# 
# Solution 1 works better we can use this trick to improve our house prediction.

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

def get_mae(X, y):
    return -1 * cross_val_score(RandomForestRegressor(50),
                                X,
                                y,
                                scoring='neg_mean_absolute_error').mean()

predictors_without_categoricals = train_predictors.select_dtypes(exclude=['object'])
mae_without_categoricals = get_mae(predictors_without_categoricals, target)
mae_one_hot_encoded = get_mae(one_hot_encoded_training_predictors, target)

print("Mean absolute error when dropping categoricals: " + str(int(mae_without_categoricals)))
print("Mean absolute error with one hot encoding: " + str(int(mae_one_hot_encoded)))


# Ensure test data is encoded in the same manner as training data with **align** command.
# 
# `join='left'` works the same way as SQL's left join, the same is true for `join='inner'`.

# In[ ]:


one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                    join='left',
                                                                    axis=1)


# Let's put what we've just learnt into practice and improve our previous submission.
# - add imputation
# - handle of categorical data with one hot encoding

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
target = train_data.SalePrice

# drop unnecessary columns
candidate_train_predictors = train_data.drop(['Id', 'SalePrice'], axis=1)
candidate_test_predictors = test_data.drop(['Id'], axis=1)

# only consider low cardinality and numeric columns for now
low_card_cols = [cname for cname in candidate_train_predictors.columns if 
                     candidate_train_predictors[cname].nunique() < 10 and
                     candidate_train_predictors[cname].dtype == "object"]
numeric_cols = [cname for cname in candidate_train_predictors.columns if
                   candidate_train_predictors[cname].dtype in ['int64', 'float64']]
my_cols = low_card_cols + numeric_cols

train_predictors = candidate_train_predictors[my_cols]
test_predictors = candidate_test_predictors[my_cols]

# one hot encoding
one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)

# we need to align after doing one hot encoding
aligned_train, aligned_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                    join='left', 
                                                                    axis=1)

# imputation
my_imputer = SimpleImputer()
final_train_predictors = my_imputer.fit_transform(aligned_train)
final_test_predictors = my_imputer.transform(aligned_test)

# train model
my_model = RandomForestRegressor(50)
my_model.fit(final_train_predictors, target)

# run model on test data
predicted_prices = my_model.predict(final_test_predictors)
# print(predicted_prices)

# prepare submission
my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('submission2.csv', index=False)


# With imputation and categorical handling, we're jumped from `81%` to `53%`, amazing!

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
target = train_data.SalePrice

# drop unnecessary columns
candidate_train_predictors = train_data.drop(['Id', 'SalePrice'], axis=1)
candidate_test_predictors = test_data.drop(['Id'], axis=1)

# only consider low cardinality and numeric columns for now
low_card_cols = [cname for cname in candidate_train_predictors.columns if 
                     candidate_train_predictors[cname].nunique() < 10 and
                     candidate_train_predictors[cname].dtype == "object"]
numeric_cols = [cname for cname in candidate_train_predictors.columns if
                   candidate_train_predictors[cname].dtype in ['int64', 'float64']]
my_cols = low_card_cols + numeric_cols

train_predictors = candidate_train_predictors[my_cols]
test_predictors = candidate_test_predictors[my_cols]

# one hot encoding
one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)

# we need to align after doing one hot encoding
aligned_train, aligned_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                    join='left', 
                                                                    axis=1)

# imputation
my_imputer = SimpleImputer()
final_train_predictors = my_imputer.fit_transform(aligned_train)
final_test_predictors = my_imputer.transform(aligned_test)

# split as it's required by XGBoost
train_X, test_X, train_y, test_y = train_test_split(final_train_predictors, 
                                                    target,
                                                    test_size=0.25)

# using XGBoost
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(train_X, train_y, early_stopping_rounds=10,
             eval_set=[(test_X, test_y)], verbose=False)

# evaluation
predictions = my_model.predict(test_X)
print("Mean aboslute error: " + str(mean_absolute_error(predictions, test_y)))

# run model on test data
predicted_prices = my_model.predict(final_test_predictors)

# prepare submission
my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('submission3.csv', index=False)


# With the help of XGBoost we're now at **38%** (1801/4755). Pretty significant improvement!

# ### Partial Dependence Plots (feature engineering, skipped)

# ### Convert to using pipelines & cross validation
# * Pipelines makes the code cleaner and easier to change.
# * Cross validation gives better estimate of the model than simple train-test-split, though it takes longer to run.

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor


train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
target = train_data.SalePrice

# drop unnecessary columns
candidate_train_predictors = train_data.drop(['Id', 'SalePrice'], axis=1)
candidate_test_predictors = test_data.drop(['Id'], axis=1)

# only consider low cardinality and numeric columns for now
low_card_cols = [cname for cname in candidate_train_predictors.columns if 
                     candidate_train_predictors[cname].nunique() < 10 and
                     candidate_train_predictors[cname].dtype == "object"]
numeric_cols = [cname for cname in candidate_train_predictors.columns if
                   candidate_train_predictors[cname].dtype in ['int64', 'float64']]
my_cols = low_card_cols + numeric_cols

train_predictors = candidate_train_predictors[my_cols]
test_predictors = candidate_test_predictors[my_cols]

# one hot encoding
one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)

# we need to align after doing one hot encoding
aligned_train, aligned_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                        join='left', 
                                                                        axis=1)

# making pipeline
my_pipeline = make_pipeline(SimpleImputer(), RandomForestRegressor())
scores = cross_val_score(my_pipeline, aligned_train, target, scoring='neg_mean_absolute_error')

# imputation
# my_imputer = SimpleImputer()
# final_train_predictors = my_imputer.fit_transform(aligned_train)
# final_test_predictors = my_imputer.transform(aligned_test)

# split as it's required by XGBoost
# train_X, test_X, train_y, test_y = train_test_split(final_train_predictors, 
#                                                     target,
#                                                     test_size=0.25)

# using XGBoost
# my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
# my_model.fit(train_X, train_y, early_stopping_rounds=10,
#              eval_set=[(train_X, train_y)], verbose=False)

# evaluation
# predictions = my_model.predict(test_X)
print("Mean aboslute error: %2f " %(-1 * scores.mean()))



# run model on test data
my_pipeline.fit(aligned_train, target)
predicted_prices = my_pipeline.predict(aligned_test)
print(predicted_prices)

# # prepare submission
# my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predicted_prices})
# my_submission.to_csv('submission3.csv', index=False)

