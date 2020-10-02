#!/usr/bin/env python
# coding: utf-8

# # Introduction
# **This workspace is built following Kaggle's Machine Learning education track.**
# 
# We follow the tutorial to build and continually improve a model to predict housing prices.
# 
# The data from the tutorial, the Melbourne data, is not available in this workspace.  We translate the concepts to work with the data in this notebook, the Iowa data.
# 
# Come to the [Learn Discussion](https://www.kaggle.com/learn-forum) forum for any questions or comments. 
# 
# # Machine Learning Tutorial
# 
# 

# # *Level 1*

# ** 2 - Starting your Machine Learning Project**
# 
# We will now initialize a Machine Learning project, load a dataset and see its description

# In[ ]:


# 2.1 - Importing a dataset using pandas

import pandas as pd

main_file_path = '../input/train.csv'
data = pd.read_csv(main_file_path)


# In[ ]:


# 2.2 - Obtaining the description of the dataset 'data'

print(data.describe())


# ** 3 - Selecting and Filtering Data**
# 
# We will now see how we can extract information from our dataset and present it

# In[ ]:


# 3.1 - List the titles of all the columns of the dataset

print(data.columns)


# In[ ]:


# 3.2 - Show the first 5 elements of a column of the dataset

data_sale_price = data.SalePrice
print(data_sale_price.head())


# In[ ]:


# 3.3 - Selecting multiple columns

# We select the two columns 'YrSold' and 'SalePrice' and show their description
# Notice that in this case we didn't use the 'print' instruction
columns_chosen = ['YrSold', 'SalePrice']
two_columns_chosen = data[columns_chosen]
two_columns_chosen.describe()


# ** 4 - First Scikit-Learn Model**
# 
# We will now select our target, predictors and create a Decision Tree Regressor Model

# In[ ]:


# 4.1 - Selecting target and predictor data

# In this step we set up some of the data in X and y so that we can use it in a predictor model
# In X we include all the fields we use as predictos
# In y we select the field we want to predict, our target
y = data.SalePrice

predictor_fields = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = data[predictor_fields]


# In[ ]:


# 4.2 - Applying the Decision Tree Regressor and using it to get a fit

# We import the 'DecisionTreeRegressor' from 'sklearn.tree'
# Then, we use it to define our model to predict y using X
from sklearn.tree import DecisionTreeRegressor

my_model = DecisionTreeRegressor()
my_model.fit(X, y)


# In[ ]:


# 4.3 - Using our model to get some predictions

# We show the initial values of the X data
# Then we show the predictions of y for those values of X
print("We will now make some predictions of house prices for Iowa data:")
print("For the given data:")
print(X.head())
print("the prices predictions are:")
print(my_model.predict(X.head()))


# ** 5 - Model Validation **
# 
# We will now split our data in train and validation sets and then use Mean Absolute Error to get an estimate of our model's accuracy

# In[ ]:


# 5.1 - Splitting data and computing mean absolute error

# We will now split our data in train and validation sets
# Using this split data, we will use this to compute the mean absolute error of our prediction
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Splitting of the data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# Construction of the model using the training data
my_model_2 = DecisionTreeRegressor()
my_model_2.fit(train_X, train_y)

# Computation of the predicted values y corresponding to the validation set X
val_predictions = my_model_2.predict(val_X)

# Computation of the mean absolute error by comparing the actua values of y in the validation set, val_y, and the
# values we computed using our model, val_predictions
print(mean_absolute_error(val_y, val_predictions))


# ** 6 - Underfitting, Overfitting and Model Optimization **
# 
# We will now define a function to compute MAE and then see how the maximum number of leaf nodes we set influences the accuracy of our model

# In[ ]:


# 6.1 - Function to compute MAE

# Definition of a function that computes the mean absolute error of a Decision Tree Regressor model for a given number of
# maximum leaf nodes, training predictors, validation predictors, training targets and validation targets
def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)


# In[ ]:


# 6.2 - Testing our model for different maximum number of leaf nodes

# We use the function mae to find out what would be a suitable number of leaf nodes to be used in our model
# This allows us to have a number of leaf nodes that neither overfits nor underfits our model
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d \t \t Mean Absolute Error: %d" %(max_leaf_nodes, my_mae))


# ** 7 - Random Forests **
# 
# We will now use a Random Forest Regressor as a model and use it to make predictions

# In[ ]:


# 7.1 - Applying the Random Forest Regressor as our model

# We will now use the Random Forest model
# We want to see if we get a better prediction than the best prediction we got using the Decision Tree model
from sklearn.ensemble import RandomForestRegressor

my_model_3 = RandomForestRegressor()
my_model_3.fit(train_X, train_y)

forest_pred = my_model_3.predict(val_X)
print(mean_absolute_error(val_y, forest_pred))


# ** 8 - Submitting From a Kernel **
# 
# We will now use our model to predict prices for the test data and then submit it

# In[ ]:


# 8.1 - Load the test data and predict its corresponding prices

test = pd.read_csv('../input/test.csv')

test_X = test[predictor_fields]

predicted_prices = my_model_2.predict(test_X)

print(predicted_prices)


# In[ ]:


# 8.2 - We will now submit the file

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

my_submission.to_csv('submission.csv', index=False)


# # *Level 2*

# ** 1 - Handling Missing Values **
# 
# We will now use different approaches to deal with columns that have missing values

# In[ ]:


# 1.1 - Basic set-up of the problem for our Iowa data

import pandas as pd

iowa_data = pd.read_csv('../input/train.csv')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

print(iowa_data.columns)
print(iowa_data.columns.size)

# Definition of the target and predictors in our model
# In the predictors we drop 'SalePrice' because it is the target and we should also drop 'Id'
iowa_target = iowa_data.SalePrice
iowa_predictors = iowa_data.drop(['SalePrice'], axis = 1)

# We add this variable which contains only the numeric predictors to give us the option of using it to create a simpler model
iowa_numeric_predictors = iowa_predictors.select_dtypes(exclude = ['object'])
print(iowa_numeric_predictors.columns)
print(iowa_numeric_predictors.columns.size)


# In[ ]:


# 1.2 - Function to measure quality of an approach

# Splitting of the data
train_X, test_X, train_y, test_y = train_test_split(iowa_numeric_predictors, iowa_target, 
                                                  train_size = 0.7, test_size = 0.3, random_state = 0)

def score_dataset(train_X, test_X, train_y, test_y):
    my_model_n = RandomForestRegressor()
    my_model_n.fit(train_X, train_y)
    predictions_n = my_model_n.predict(test_X)
    return mean_absolute_error(test_y, predictions_n)


# In[ ]:


# 1.3 - Dropping Columns with Missing Values

cols_with_missing = [col for col in train_X.columns
                            if train_X[col].isnull().any()]
print(cols_with_missing)
print(len(cols_with_missing))
print(iowa_numeric_predictors.columns)
print(iowa_numeric_predictors.columns.size)
reduced_train_X = train_X.drop(cols_with_missing, axis = 1)
reduced_train_X_no_Id = reduced_train_X.drop(['Id'], axis = 1)
reduced_test_X = test_X.drop(cols_with_missing, axis = 1)
reduced_test_X_no_Id = reduced_test_X.drop(['Id'], axis = 1)
print(reduced_train_X.columns)
print(reduced_train_X.columns.size)
print(reduced_train_X_no_Id.columns)
print(reduced_train_X_no_Id.columns.size)
print("The Mean Absolute Error from Dropping Columns with missing values is: %d" 
      %(score_dataset(reduced_train_X, reduced_test_X, train_y, test_y)))
print("The Mean Absolute Error from Dropping Columns with missing values and columns 'Id' is: %d" 
      %(score_dataset(reduced_train_X_no_Id, reduced_test_X_no_Id, train_y, test_y)))


# In[ ]:


# 1.4 - Imputation

from sklearn.preprocessing import Imputer

my_imputer = Imputer()
imputed_train_X = my_imputer.fit_transform(train_X)
imputed_test_X = my_imputer.transform(test_X)
print("The Mean Absolute Error from Imputation is: %d" 
      %(score_dataset(imputed_train_X, imputed_test_X, train_y, test_y)))


# In[ ]:


# 1.5 - Imputation with extra Columns Showing what was Imputed

imputed_train_X_plus = train_X.copy()
imputed_test_X_plus = test_X.copy()
print(imputed_train_X_plus.columns)
cols_with_missing = [col for col in train_X.columns
                            if train_X[col].isnull().any()]

# Procedure to add columns identifying missing elements
for col in cols_with_missing:
    imputed_train_X_plus[col + '_was_missing'] = imputed_train_X_plus[col].isnull()
    imputed_test_X_plus[col + '_was_missing'] = imputed_test_X_plus[col].isnull()

# Imputation
# Here we are inputing in the data set that has the added columns '_was_missing'
my_imputer_2 = Imputer()
imputed_train_X_plus = my_imputer_2.fit_transform(imputed_train_X_plus)
imputed_test_X_plus = my_imputer_2.transform(imputed_test_X_plus)


print('The Mean Absolute Error from Imputation Tracking what was Imputed is: %d'
         %(score_dataset(imputed_train_X, imputed_test_X, train_y, test_y)))


# ** 2 - Using Categorical Data with One Hot Encoding**
# 
# We will now use One-Hot Encoding to deal with Categorical Data

# In[ ]:


# 2.1 - Basic set-up of the problem for our Iowa data

import pandas as pd

# Importing train and test data
iowa_train_data = pd.read_csv('../input/train.csv')
iowa_test_data = pd.read_csv('../input/test.csv')

# We drop houses (which are lines of the table) where the entry 'SalePrice' is missing
# This is because if there is no target information, the remaining information doesn't help building the model
iowa_train_data.dropna(axis = 0, subset = ['SalePrice'], inplace = True)

iowa_target = iowa_train_data.SalePrice

# Like in 1.3 we drop columns where there are missing values 
cols_with_missing = [col for col in iowa_train_data.columns
                            if iowa_train_data[col].isnull().any()]
print(cols_with_missing)
print(len(cols_with_missing))

candidate_train_predictors = iowa_train_data.drop(['Id', 'SalePrice'] + cols_with_missing, axis = 1)
candidate_test_predictors = iowa_test_data.drop(['Id'] + cols_with_missing, axis = 1)
print(candidate_train_predictors.columns)
print(len(candidate_train_predictors.columns))
print(candidate_test_predictors.columns)
print(len(candidate_test_predictors.columns))

# Consider the number of different elements of the type 'object' that exist in columns that have non-numeric values
# We will now find the columns with low cardinality (of these elements) to which One Hot Encoding will be applied
low_cardinality_cols = [col_2 for col_2 in candidate_train_predictors.columns
                                if candidate_train_predictors[col_2].nunique() < 10
                                and candidate_train_predictors[col_2].dtype == 'object']
print(len(low_cardinality_cols))

# Print all the columns of the type 'object' that have non-numeric values
low_cardinality_cols2 = [col_2 for col_2 in candidate_train_predictors.columns if candidate_train_predictors[col_2].dtype == 'object']
print(len(low_cardinality_cols2))

numeric_cols = [col_3 for col_3 in candidate_train_predictors.columns
                        if candidate_train_predictors[col_3].dtype in ['int64', 'float64']]
print(len(numeric_cols))

my_cols = low_cardinality_cols + numeric_cols
print(len(my_cols))

# The columns we want to use are those that were selected in low_cardinality_cols and numeric_cols
# We can now apply this in our candidate_train_predictors and select only these relevant columns
train_predictors = candidate_train_predictors[my_cols]
test_predictors = candidate_test_predictors[my_cols]


# In[ ]:


# 2.2 - We can see the data type of each column

print(train_predictors.dtypes)
print(len(train_predictors.columns))


# In[ ]:


# 2.3 - Apply One-Hot Encoding to our columns

# Applying One-Hot Encoding to training predictors and test predictors
one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)

# To ensure that these two sets were One-Hot encoded in the same order, we use the following command
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors, join = 'left', axis = 1)
# Instead of using "join = 'left'", we could have used "join = 'inner'" so that only the columns appearing in both
# datasets would be kept

# We can now see the columns that appear as the result of One-Hot Encoding
print(final_train.dtypes)


# In[ ]:


# 2.4 - Compute MAE of the model with One-Hot Encoded categoricals and comparison with model with dropped categoricals

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

def get_mae(X, y):
    return -1 * cross_val_score(RandomForestRegressor(), X, y, scoring = 'neg_mean_absolute_error').mean()

predictors_without_categoricals = train_predictors.select_dtypes(exclude = ['object'])

mae_without_categoricals = get_mae(predictors_without_categoricals, iowa_target)

mae_one_hot_encoded = get_mae(final_train, iowa_target)

print('Mean Absolute Error dropping categoricals: %d' %(mae_without_categoricals))
print('Mean Absolute Error with One-Hot Encoding: %d' %(mae_one_hot_encoded))


# ** 3 - Learning to use XGBoost **
# 
# We will now apply XGBoost to our model

# In[ ]:


# 3.1 - Basic set-up of the problem for our Iowa data

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

# Importing train data
iowa_data = pd.read_csv('../input/train.csv')

# We drop houses (which are lines of the table) where the entry 'SalePrice' is missing
# This is because if there is no target information, the remaining information doesn't help building the model
iowa_data.dropna(axis = 0, subset = ['SalePrice'], inplace = True)

# We define the target data to be 'SalePrice'
y = iowa_data.SalePrice
# The predicotr data will be all the numeric fields except for 'SalePrice'
X = iowa_data.drop(['SalePrice'], axis = 1).select_dtypes(exclude = ['object'])

# We split our data into two sets: training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size = 0.25)

my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
val_X = my_imputer.transform(val_X)


# In[ ]:


# 3.2 - Applying the XGBoost model

from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators = 50000, learning_rate = 0.05)

my_model.fit(train_X, train_y, early_stopping_rounds = 5, eval_set=[(val_X, val_y)], verbose = False)


# In[ ]:


# 3.3 - Using the XGBoost model to make predictions

predictions = my_model.predict(val_X)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, val_y)))


# **4 - Partial Dependence Plots**
# 
# We will now do some partial dependence plots for our model

# In[ ]:


# 4.1 - Finding out which variables are in our data so that we can decide which ones to plot

import pandas as pd

# Importing train data
iowa_data = pd.read_csv('../input/train.csv')

# We need to know the names of the columns in our data so that we can find out their type
print(iowa_data.columns)

# We are only interested in columns that have a numeric type
print(iowa_data.SalePrice.dtypes)
print(iowa_data.LotArea.dtypes)
print(iowa_data.YearBuilt.dtypes)
print(iowa_data.OverallQual.dtypes)


# In[ ]:


# 4.2 - We will now get the data for our problem and print the plots

from sklearn.preprocessing import Imputer
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence

# We define a function that returns us some target data and predictors
def get_some_data():
    iowa_data_2 = pd.read_csv('../input/train.csv')
    cols_to_use = ['LotArea', 'YearBuilt', 'OverallQual']
    
    y = iowa_data_2.SalePrice
    X = iowa_data_2[cols_to_use]
    
    my_imputer = Imputer()
    imputed_X = my_imputer.fit_transform(X)
    return imputed_X, y

# We begin by using our function to obtain some data
X, y = get_some_data()

# We now fit our model for the problem
my_model_p = GradientBoostingRegressor()
my_model_p.fit(X, y)

# We now make the partial dependence plots
my_model_plots = plot_partial_dependence(my_model_p, 
                                         features = [0, 1, 2], 
                                         X = X, 
                                         feature_names = ['LotArea', 'YearBuilt', 'OverallQual'], 
                                         grid_resolution = 100)


# **5 - Pipelines**
# 
# We will now apply Pipelines to our problem

# In[ ]:


# 5.1 - Importing data and pre-processing by using One-Hot Encoding

import pandas as pd
from sklearn.model_selection import train_test_split

iowa_data_5 = pd.read_csv('../input/train.csv')

y = iowa_data_5.SalePrice
X = iowa_data_5.drop(['SalePrice'], axis = 1)

train_X, test_X, train_y, test_y = train_test_split(X, y)

# Consider the number of different elements of the type 'object' that exist in columns that have non-numeric values
# We will now find the columns with low cardinality (of these elements) to which One Hot Encoding will be applied
low_cardinality_cols = [col for col in train_X.columns
                                if train_X[col].nunique() < 10
                                and train_X[col].dtype == 'object']

numeric_cols = [col2 for col2 in train_X.columns
                        if train_X[col2].dtype in ['int64', 'float64']]

my_cols = low_cardinality_cols + numeric_cols

# The columns we want to use are those that were selected in low_cardinality_cols and numeric_cols
# We can now apply this in our candidate_train_predictors and select only these relevant columns
train_predictors = train_X[my_cols]
test_predictors = test_X[my_cols]

# Applying One-Hot Encoding to training predictors and test predictors
one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)

# To ensure that these two sets were One-Hot encoded in the same order, we use the following command
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors, join = 'left', axis = 1)
# Instead of using "join = 'left'", we could have used "join = 'inner'" so that only the columns appearing in both
# datasets would be kept


# In[ ]:


# 5.2 - Applying Pipelines

from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline

from sklearn.metrics import mean_absolute_error

iowa_pipeline = make_pipeline(Imputer(), RandomForestRegressor())

iowa_pipeline.fit(final_train, train_y)
predictions = iowa_pipeline.predict(final_test)

print(mean_absolute_error(test_y, predictions))


# **6 - Cross-Validation**
# 
# We will now use Cross-Validation to build our model

# In[ ]:


# 6.1 - We begin by importing some data and choosing a set of columns to use

import pandas as pd

iowa_data = pd.read_csv('../input/train.csv')

cols_to_use = ['LotArea', 'YearBuilt', 'OverallQual', 'OverallCond', 'Fireplaces']

X = iowa_data[cols_to_use]
y = iowa_data.SalePrice


# In[ ]:


# 6.2 - We apply a pipeline of our modeling steps

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor

my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())


# In[ ]:


# 6.3 - Compute cross-validation scores

from sklearn.model_selection import cross_val_score

scores = cross_val_score(my_pipeline, X, y, scoring = 'neg_mean_absolute_error')
print(scores)

print('Mean Absolute Error: %2f' %(-1 * scores.mean()))


# In[ ]:


# 6.4 - We now repeat the process without some predictors
# As a result of this procedure we expect to get a bigger mean absolute error

cols_to_use_2 = ['OverallQual', 'OverallCond', 'Fireplaces']

X_2 = iowa_data[cols_to_use_2]
y_2 = iowa_data.SalePrice

my_pipeline_2 = make_pipeline(Imputer(), RandomForestRegressor())

scores_2 = cross_val_score(my_pipeline_2, X_2, y_2, scoring = 'neg_mean_absolute_error')
print(scores_2)

print('Mean Absolute Error: %2f' %(-1 * scores_2.mean()))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




