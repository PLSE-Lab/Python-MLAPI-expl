#!/usr/bin/env python
# coding: utf-8

# # Learn Machine Learning
# This notebook will guide you to kick start with you first Machine Learning Model.
# You can apply these techniques in other datasets for your practice. (Ex: Titanic Dataset)
# 
# > **If this helps you in anyways, then Please Upvote.**

# In[ ]:


import pandas as pd
# Read CSV file using pandas library
main_file_path = '../input/house-prices-advanced-regression-techniques/train.csv' # this is the path to the Iowa data that you will use
data = pd.read_csv(main_file_path)

# Run this code block with the control-enter keys on your keyboard. Or click the blue botton on the left
print(data.describe())


# In[ ]:


#This shows you the list of columns in a DataFrame
print (data.columns)


# In[ ]:


#Print values of a particular columns.
price = data.SalePrice
# head() returns the first few values as given.
print (price.head(5))


# In[ ]:


# To print Multiple columns values
lot_price = ['Id','LotArea','SalePrice']
print (data[lot_price].head(5))


# # Building the first model

# 1.  Choose **predicting label y**.
# 2. Choose the **features x**.

# In[ ]:


y = data.SalePrice
features = ['Id','LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X = data[features]


# # Using Decision Tree Classifier
# 
# You can choose any model and run the dataset.

# In[ ]:


from sklearn.tree import DecisionTreeRegressor

# Define model
iowa_model = DecisionTreeRegressor()

# Fit model
iowa_model.fit(X, y)


# In[ ]:


print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(iowa_model.predict(X.head()))


# # Accuracy Check

# In[ ]:


from sklearn.model_selection import train_test_split

# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
# Define model
iowa_model = DecisionTreeRegressor()
# Fit model
iowa_model.fit(train_X, train_y)


# In[ ]:


from sklearn.metrics import mean_absolute_error

predicted_home_prices = iowa_model.predict(val_X)
mean_absolute_error(val_y, predicted_home_prices)


# In[ ]:


def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)


# In[ ]:


# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# # Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

model = RandomForestRegressor()
model.fit(train_X, train_y)
preds = model.predict(val_X)
rf_score = mean_absolute_error(val_y, preds)
print (rf_score)


# # Submission
# This is the way to submit your results in any of the kaggle competitions.

# In[ ]:


my_submission = pd.DataFrame({'Id': val_X.Id, 'SalePrice': preds})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)


# # Data Wrangling
# Data wranling is needed because a large dataset may contains many null values in it. These null values can make your model poor. So, there are three solutions to overcome those null values

# In[ ]:


# The below DataFrame contains null values
df_predictors = data.drop(['SalePrice'],axis=1)
df_target = data['SalePrice']

# For the sake of keeping the example simple, we'll use only numeric predictors. 
#This will remove the fence data
df_numeric_predictors = df_predictors.select_dtypes(exclude=['object'])

X_train, X_test, y_train, y_test = train_test_split(df_numeric_predictors, 
                                                    df_target,
                                                    train_size=0.7, 
                                                    test_size=0.3,
                                                    random_state=0)
# We are using here random forest model.
def random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)


# ##  Compare the Mean Absolute Error of each solution.

# ## Soln 1 : Drop columns with missing values

# In many cases, you'll have both a training dataset and a test dataset. You will want to drop the same columns in both DataFrames. In that case, you would write

# In[ ]:


cols_with_missing = [col for col in X_train.columns 
                                 if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_test  = X_test.drop(cols_with_missing, axis=1)


# In[ ]:


print("Mean Absolute Error from dropping columns with Missing Values:")
print(random_forest(reduced_X_train, reduced_X_test, y_train, y_test))


# If some columns has some useful values i.e the columns with missing values. Then your model may result in an error.
# 
# So, this is not always the best solution. However it can be helpful when most values in the columns are missing.

# ## Soln 2 : Imputation
# Imputation fills in the missing value with some number i.e the mean value. The imputed value won't be exactly right in most cases, but it usually gives more accurate models than dropping the column entirely.

# In[ ]:


from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)
print("Mean Absolute Error from Imputation:")
print(random_forest(imputed_X_train, imputed_X_test, y_train, y_test))


# ## Soln 3 : An Extension To Imputation

# In[ ]:


imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()

cols_with_missing = (col for col in X_train.columns 
                                 if X_train[col].isnull().any())
for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

print("Mean Absolute Error from Imputation while Track What Was Imputed:")
print(random_forest(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))


# In some cases this approach will meaningfully improve results. In other cases, it doesn't help at all.
# __________________________________________________________________________________________________________________________________________________________

# # Categorize the categorical data

# Since string values cant be processed by any model, we convert the categorical data to numerical data using One Hot Encoding.
# 
# One Hot Encoding creates new columns and store the binary data on that particular columns to represent the presence of original data.
# 
# For Example : A gender column has two categorical value 'Male' and 'Female'. So using One Hot Encoding it create two columns named 'Male' and 'Female' and store 0 and 1 based on the presence of original data.

# In[ ]:


train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

# Drop houses where the target is missing
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)

target = train_data.SalePrice
#Contains all the missing values columns name
cols_with_missing = [col for col in train_data.columns 
                                 if train_data[col].isnull().any()]

candidate_train_predictors = train_data.drop(['Id', 'SalePrice'] + cols_with_missing, axis=1)
candidate_test_predictors = test_data.drop(['Id'] + cols_with_missing, axis=1)


# In[ ]:


# "cardinality" means the number of unique values in a column.
# We use it as our only way to select categorical columns here. This is convenient, though
# a little arbitrary.
low_cardinality_cols = [cname for cname in candidate_train_predictors.columns if 
                                candidate_train_predictors[cname].nunique() < 10 and
                                candidate_train_predictors[cname].dtype == "object"]
numeric_cols = [cname for cname in candidate_train_predictors.columns if 
                                candidate_train_predictors[cname].dtype in ['int64', 'float64']]
my_cols = low_cardinality_cols + numeric_cols
train_predictors = candidate_train_predictors[my_cols]
test_predictors = candidate_test_predictors[my_cols]
#train_predictors.dtypes.sample(20)


# It's most common to one-hot encode these "object" columns, since they can't be plugged directly into most models. Pandas offers a convenient function called get_dummies to get one-hot encodings. Call it like this:

# In[ ]:


one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)


# **Compare the MAE between**
#         1. Numerical predictors, where we drop categoricals.   
#         2. One-hot encoded categoricals as well as numeric predictors

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
# Evaluation using cross validation
def get_mae(X, y):
    # multiple by -1 to make positive MAE score instead of neg value returned as sklearn convention
    return -1 * cross_val_score(RandomForestRegressor(50), 
                                X, y, 
                                scoring = 'neg_mean_absolute_error').mean()

predictors_without_categoricals = train_predictors.select_dtypes(exclude=['object'])

mae_without_categoricals = get_mae(predictors_without_categoricals, target)

mae_one_hot_encoded = get_mae(one_hot_encoded_training_predictors, target)

print('Mean Absolute Error when Dropping Categoricals: ' + str(int(mae_without_categoricals)))
print('Mean Abslute Error with One-Hot Encoding: ' + str(int(mae_one_hot_encoded)))


# # XGBoost
# XGBoost is an implementation of the Gradient Boosted Decision Trees algorithm.
# 
# ![](https://i.imgur.com/e7MIgXk.png)
# We go through cycles that repeatedly builds new models and combines them into an ensemble model. We start the cycle by calculating the errors for each observation in the dataset. We then build a new model to predict those. We add predictions from this error-predicting model to the "ensemble of models."
# 
# To make a prediction, we add the predictions from all previous models. We can use these predictions to calculate new errors, build the next model, and add it to the ensemble.
# 
# There's one piece outside that cycle. We need some base prediction to start the cycle. In practice, the initial predictions can be pretty naive. Even if it's predictions are wildly inaccurate, subsequent additions to the ensemble will address those errors.
# 
# This process may sound complicated, but the code to use it is straightforward. We'll fill in some additional explanatory details in the model tuning section below.

# In[ ]:


from sklearn.preprocessing import Imputer

data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)

my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)


# In[ ]:


from xgboost import XGBRegressor

# my_model = XGBRegressor()
# # Add silent=True to avoid printing out updates with each cycle
# my_model.fit(train_X, train_y, verbose=False)

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.03)
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)

# make predictions
predictions = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))


# # Partial Dependence Plot
# 
# We'll start with 2 partial dependence plots showing the relationship (according to our model) between Price and a couple variables from the Housing dataset. We'll walk through how these plots are created and interpreted.

# In[ ]:


import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.preprocessing import Imputer

cols_to_use = ['LotArea','YearBuilt','1stFlrSF']

def get_some_data():
    data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
    y = data.SalePrice
    X = data[cols_to_use]
    my_imputer = Imputer()
    imputed_X = my_imputer.fit_transform(X)
    return imputed_X, y
    
X, y = get_some_data()
my_model = GradientBoostingRegressor()
my_model.fit(X, y)
my_plots = plot_partial_dependence(my_model,
                                   features=[0,2],
                                   X=X,
                                   feature_names=cols_to_use, 
                                   grid_resolution=10)


# The left plot shows the partial dependence between our target, Sales Price, and the Lot Area.
# 
# **The partial dependence plot is calculated only after the model has been fit.**

# # Cross Validation
# 
# The diagram below shows an example of the training subsets and evaluation subsets generated in k-fold cross-validation. Here, we have total 25 instances. In first iteration we use the first 20 percent of data for evaluation, and the remaining 80 percent for training([1-5] testing and [5-25] training) while in the second iteration we use the second subset of 20 percent for evaluation, and the remaining three subsets of the data for training([5-10] testing and [1-5 and 10-25] training), and so on.
# 
# ![](https://cdncontribute.geeksforgeeks.org/wp-content/uploads/crossValidation.jpg)
# 
# **Unlike train_test_splits it does not divide the training and testing dataset into a given percentage.**

# In[ ]:


data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())

from sklearn.model_selection import cross_val_score
scores = cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error')
print(scores)

print('Mean Absolute Error %2f' %(-1 * scores.mean()))


# So you can see that how the Mean Accuracy Error decresed gradually by applying different techniques.
# Try this on other dataset and see the results.
# 
# Comment if you have any doubts or you can give any suggestions to improve this notebook.
# > **If this notebook helped you in anyways, then Please Upvote.**

# In[ ]:




