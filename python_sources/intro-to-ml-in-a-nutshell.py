#!/usr/bin/env python
# coding: utf-8

# # Acknowledgements
# > Most information in this notebook are based on Kaggle micro-courses available in [Kaggle Learn](https://www.kaggle.com/learn/overview).

# # Terminology

# - **Fitting/ Training the model:** the process of capturing patterns from data.  
# - **Training data:** the data used to fit the model.
# - **Prediction target:** the column we want to predict, called **y** by convention.
# - **Features:** the columns that are inputted into the model and later used to make predictions, called **X** by convention.
# - **Validation data:** the data excluded from the model-building process, and then used to test the model's accuracy.
# - **Tree's depth:** a measure of how many splits the tree makes before coming to a prediction.
# - **Overfitting:** capturing spurious patterns that won't recur in the future, leading to less accurate predictions. 
# - **Underfitting:** failing to capture relevant patterns, leading to less accurate predictions.
# - **CSV file:** a table of values separated by commas. Hence the name: "Comma-Separated Values".
# - **Ensemble method:** a method that combines the predictions of several models (e.g., several trees, in the case of random forests).
# - **Gradient boosting**: a method that goes through cycles to iteratively add models into an ensemble.
# - **Target leakage:** a data leakage that occurs when your predictors include data that will not be available at the time you make predictions. 
# - **Train-Test Contamination:** a data leakage that occurs if the validation data affects the preprocessing behavior.

# # Application 1 | Model Building

# The steps to building and using a model are:
# 
# - **Define:** what type of model will it be? Some other parameters of the model type are specified too.
# - **Fit:** capture patterns from provided data. This is the heart of modeling.
# - **Predict:** just what it sounds like
# - **Evaluate:** determine how accurate the model's predictions are.

# In[ ]:


# # # # # # # IMPORTS # # # # # # #

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# # # # # # # GETTING DATA # # # # # # #

home_data_file_path = '../input/home-data-for-ml-course/train.csv' # save the path of the file to read 
home_data = pd.read_csv(home_data_file_path, index_col='Id') # read the data and store it in DataFrame titled home_data


# # # # # # # EXPLORING DATA # # # # # # #

obj_cols = [col for col in home_data.columns if home_data[col].dtype == 'object'] 
num_cols = [col for col in home_data.columns if home_data[col].dtype != 'object'] 
missing_val_cols = [col for col in home_data.columns if home_data[col].isnull().any()] 

print("Home Data | #rows: {}, #columns: {} (#numerical: {}, #categorical: {}), #columns with missing values: {}\n".format(home_data.shape[0], home_data.shape[1], len(num_cols), len(obj_cols), len(missing_val_cols)))


# # # # # # # DATA MANIPULATION # # # # # # #

# SELECT TARGET #
y = home_data.SalePrice # we use the dot notation to select the column we want to predict 

# SELECT FEATURES #
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd'] # to keep things simple, we choose only 7 features out of 80 available, all numerical
X = home_data[features] 

# SPLIT DATA INTO TRAINING AND VALIDAION SETS #
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0) # supplying a numeric value to the random_state argument guarantees we get the same split every time we run this script

print("Training Data | #rows: {}, #features: {}".format(X_train.shape[0], X_train.shape[1]))
print("Validation Data | #rows: {}, #features: {}\n".format(X_valid.shape[0], X_valid.shape[1]))


# # # # # # # DECESION TREE MODEL # # # # # # #

# DEFINE #
decision_tree_model = DecisionTreeRegressor(random_state=0) # many machine learning models allow some randomness in model training. Specifying a number for random_state ensures you get the same results in each run.
# FIT #
decision_tree_model.fit(X_train, y_train) # fit the Model
# PREDICT #
val_predictions = decision_tree_model.predict(X_valid) # make validation predictions
# EVALUATE
val_mae = mean_absolute_error(val_predictions, y_valid) # calculate mean absolute error

print("Validation MAE | Decision Tree Model (when not specifying max_leaf_nodes) -> {:,.0f}".format(val_mae))


# Define a function to calculate MAE scores from different values for max_leaf_nodes
def get_mae(max_leaf_nodes):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(X_train, y_train)
    val_predictions = model.predict(X_valid)
    return mean_absolute_error(y_valid, val_predictions)

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
scores = {leaf_size: get_mae(leaf_size) for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key=scores.get)

decision_tree_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1) 
decision_tree_model.fit(X_train, y_train)
val_predictions = decision_tree_model.predict(X_valid)
val_mae = mean_absolute_error(val_predictions, y_valid)

print("Validation MAE | Decision Tree Model (when best value of max_leaf_nodes '{}' is chosen) -> {:,.0f}".format(best_tree_size, val_mae) + "\n")


# # # # # # # RANDOM FOREST MODEL # # # # # # #

random_forest_model = RandomForestRegressor(random_state=0)
random_forest_model.fit(X_train, y_train)
val_predictions = random_forest_model.predict(X_valid)
val_mae = mean_absolute_error(val_predictions, y_valid)

print("Validation MAE | Random Forest Model -> {:,.0f}".format(val_mae))


model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)
model_6 = RandomForestRegressor(n_estimators=300, criterion='mae', random_state=0)
model_7 = RandomForestRegressor(n_estimators=150, min_samples_split=10, criterion='mae', random_state=0)

models = [model_1, model_2, model_3, model_4, model_5, model_6, model_7]

# Function for comparing different models
def score_model(model):
    model.fit(X_train, y_train)
    val_predictions = model.predict(X_valid)
    return mean_absolute_error(y_valid, val_predictions)

for i in range(0, len(models)):
    val_mae = score_model(models[i])
    print("Validation MAE | Random Forest Model {} -> {:,.0f}".format(i+1, val_mae))


# # Application 2 | Missing Values

# ### Approach 1 | Drop Missing Values
# Drop columns with missing values entirely. Unless most values in the dropped columns are missing, the model loses access to a lot of (potentially useful!) information with this approach.
# ### Approach 2 | Imputation
# Fill in the missing values with some number. The imputed value won't be exactly right in most cases, but it usually leads to more accurate models than you would get from dropping the column entirely.
# ### Approach 3 | An Extension To Imputation
# Impute the missing values as before and, additionally, for each column with missing entries in the original dataset, add a new column that shows the location of the imputed entries. As imputed values may be systematically above or below their actual values (which weren't collected in the dataset). Or rows with missing values may be unique in some other way. In that case, your model would make better predictions by considering which values were originally missing. In some cases, this approach will meaningfully improve results. In other cases, it doesn't help at all.

# In[ ]:


# # # # # # # IMPORTS # # # # # # #

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# # # # # # # GETTING DATA # # # # # # #

home_data = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')


# # # # # # # DATA MANIPULATION # # # # # # #

home_data.dropna(axis=0, subset=['SalePrice'], inplace=True) # remove rows with missing target from the dataset

# SELECT TARGET #
y = home_data.SalePrice

# SELECT FEATURES #
home_data.drop(['SalePrice'], axis=1, inplace=True) # drop the target column from the dataset
X = home_data.select_dtypes(exclude=['object']) # to keep things simple, we'll use only numerical predictors

# SPLIT DATA INTO TRAINING AND VALIDAION SETS #
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)


# # # # # # # EXPLORING TRAINING DATA # # # # # # #

obj_cols = [col for col in X_train.columns if X_train[col].dtype == 'object'] 
num_cols = [col for col in X_train.columns if X_train[col].dtype != 'object'] 
missing_val_cols = [col for col in X_train.columns if X_train[col].isnull().any()] 

print("Training Data | #rows: {}, #features: {} (#numerical: {}, #categorical: {}), #features with missing values: {}\n".format(X_train.shape[0], X_train.shape[1], len(num_cols), len(obj_cols), len(missing_val_cols)))

missing_val_count_by_column = X_train.isnull().sum()

print(missing_val_count_by_column[missing_val_count_by_column > 0])

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    val_predictions = model.predict(X_valid)
    return mean_absolute_error(y_valid, val_predictions)


# # # # # # # DROP COLUMNS WITH MISSING VALUES # # # # # # #

reduced_X_train = X_train.drop(missing_val_cols, axis=1)
reduced_X_valid = X_valid.drop(missing_val_cols, axis=1)

print("\nValidation MAE | Drop columns with missing values -> {:,.0f}".format(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid)))


# # # # # # # IMPUTATION # # # # # # #

# Impute missing values with the mean value along each column
imputer = SimpleImputer(strategy='mean') 
imputed_X_train = pd.DataFrame(imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(imputer.transform(X_valid))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("Validation MAE | Imputation -> {:,.0f}".format(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid)))


# # # # # # # IMPUTATION PLUS # # # # # # #

# Make copy to avoid changing original data (when imputing)
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# Make new columns indicating what will be imputed
for col in missing_val_cols:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()
    
my_imputer = SimpleImputer(strategy='mean')
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

print("Validation MAE | Imputation Plus -> {:,.0f}".format(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid)))


# # Application 3 | Categorical Variables

# ### Approach 1 | Drop Categorical Variables
# Drop columns with categorical variables entirely. This approach will only work well if the columns did not contain useful information.
# ### Approach 2 | Label Encoding
# Assign each unique value to a different integer. This approach assumes an ordering of the categories: "Never" (0) < "Rarely" (1) < "Most days" (2) < "Every day" (3). This assumption makes sense in this example, because there is an indisputable ranking to the categories. 
# 
# Not all categorical variables have a clear ordering in the values, but we refer to those that do as **ordinal variables**. 
# 
# For tree-based models (like decision trees and random forests), you can expect label encoding to work well with ordinal variables.
# ### Approach 3 | One-Hot Encoding
# Create new columns indicating the presence (or absence) of each possible value in the original data. In contrast to label encoding, one-hot encoding does not assume an ordering of the categories. Thus, you can expect this approach to work particularly well if there is no clear ordering in the categorical data (e.g., "Red" is neither more nor less than "Yellow"). 
# 
# We refer to categorical variables without an intrinsic ranking as **nominal variables**. 
# 
# One-hot encoding generally does not perform well if the categorical variable takes on a large number of values (i.e., you generally won't use it for variables taking more than 15 different values, i.e. when cardinality is bigger then 15).
# 
# We refer to the number of unique entries of a categorical variable as the **cardinality** of that categorical variable.

# In[ ]:


# # # # # # # IMPORTS # # # # # # #

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# # # # # # # GETTING DATA # # # # # # #

home_data = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')


# # # # # # # DATA MANIPULATION # # # # # # #

home_data.dropna(axis=0, subset=['SalePrice'], inplace=True) # remove rows with missing target from the dataset

# SELECT TARGET #
y = home_data.SalePrice

# SELECT FEATURES #
home_data.drop(['SalePrice'], axis=1, inplace=True) # drop the target column from the dataset

missing_val_cols = [col for col in home_data.columns if home_data[col].isnull().any()] # get names of columns with missing values 
X = home_data.drop(missing_val_cols, axis=1) # to keep things simple, we'll drop columns with missing values

# SPLIT DATA INTO TRAINING AND VALIDAION SETS #
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)


# # # # # # # EXPLORING TRAINING DATA # # # # # # #

obj_cols = [col for col in X_train.columns if X_train[col].dtype == 'object'] 
num_cols = [col for col in X_train.columns if X_train[col].dtype != 'object'] 
missing_val_cols = [col for col in X_train.columns if X_train[col].isnull().any()] 

print("Training Data | #rows: {}, #features: {} (#numerical: {}, #categorical: {}), #features with missing values: {}\n".format(X_train.shape[0], X_train.shape[1], len(num_cols), len(obj_cols), len(missing_val_cols)))

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    val_predictions = model.predict(X_valid)
    return mean_absolute_error(y_valid, val_predictions)


# # # # # # # DROP CATEGORICAL VARIABLES # # # # # # #

drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

print("Validation MAE | Drop categorical variables -> {:,.0f}".format(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid)))


# # # # # # # LABEL ENCODING # # # # # # #

# CAUTION: Values might differ between the training and validation set, thus resulting an error when fitting different values in the validation set
print("\nUnique values in 'Condition2' column in training data:", X_train['Condition2'].unique())
print("Unique values in 'Condition2' column in validation data:", X_valid['Condition2'].unique())

# Columns that can be safely label encoded
good_label_cols = [col for col in obj_cols if set(X_train[col]) == set(X_valid[col])]

# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(obj_cols)-set(good_label_cols))

print("\n#categorical features that will be label encoded: {}, #categorical features that will be dropped: {}\n".format(len(good_label_cols), len(bad_label_cols)))

# Drop categorical columns that will not be encoded
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)

# Apply label encoder to each column with categorical data
label_encoder = LabelEncoder() 
for col in good_label_cols:
    label_X_train[col] = label_encoder.fit_transform(X_train[col])
    label_X_valid[col] = label_encoder.transform(X_valid[col])   
    
print("Validation MAE | Label Encoding -> {:,.0f}\n".format(score_dataset(label_X_train, label_X_valid, y_train, y_valid)))


# # # # # # # ONE-HOT ENCODING # # # # # # #

# Columns that will be one-hot encoded
low_cardinality_cols = [col for col in obj_cols if X_train[col].nunique() < 10]

# Columns that will be dropped from the dataset
high_cardinality_cols = list(set(obj_cols)-set(low_cardinality_cols))

print("#categorical features that will be one-hot encoded: {}, #categorical features that will be dropped: {}\n".format(len(low_cardinality_cols), len(high_cardinality_cols)))

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False) # handle_unknown='ignore' -> to avoid errors when the validation data contains classes that aren't represented in the training data; sparse=False -> ensures that the encoded columns are returned as a numpy array (instead of a sparse matrix)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols])) 
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols])) 

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (as they will be replaced with one-hot encoding)
num_X_train = X_train.drop(obj_cols, axis=1)
num_X_valid = X_valid.drop(obj_cols, axis=1)

# Add the one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

print("Validation MAE | One-Hot Encoding -> {:,.0f}".format(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid)))


# # Application 4 | Pipelining

# **Pipelines** are a simple way to keep your data **preprocessing** and **modeling** code organized. Specifically, a pipeline bundles preprocessing and modeling steps so you can use the whole bundle as if it were a single step.
# 
# ### Step 1 | Define Preprocessing Steps
# **ColumnTransformer** class bundle together different preprocessing steps.
# 
# For example: 
# - **imputes** missing values in *numerical data*, and
# - **imputes** missing values and applies a **one-hot encoding** to *categorical data*.
# 
# ### Step 2 | Define the Model
# ### Step 3 | Create and Evaluate the Pipeline
# **Pipeline** class define a pipeline that bundles the preprocessing and modeling steps. There are a few important things to notice:
# 
# - With the pipeline, we **preprocess the training data and fit the model** in a **single line of code**. (In contrast, without a pipeline, we have to do imputation, one-hot encoding, and model training in separate steps. This becomes especially messy if we have to deal with both numerical and categorical variables!)
# - With the pipeline, we supply the unprocessed features in *X_valid* to the *predict()* command, and the **pipeline automatically preprocesses the features before generating predictions**. (However, without a pipeline, we have to remember to preprocess the validation data before making predictions.)
# 
# Pipelines are valuable for cleaning up machine learning code and avoiding errors, and are especially useful for workflows with sophisticated data preprocessing.

# In[ ]:


# # # # # # # IMPORTS # # # # # # #

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# # # # # # # GETTING DATA # # # # # # #

home_data = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')


# # # # # # # DATA MANIPULATION # # # # # # #

home_data.dropna(axis=0, subset=['SalePrice'], inplace=True) # remove rows with missing target from the dataset

# SELECT TARGET #
y = home_data.SalePrice

# SELECT FEATURES #
home_data.drop(['SalePrice'], axis=1, inplace=True) # drop the target column from the dataset

obj_cols = [col for col in home_data.columns if home_data[col].nunique() < 10 and home_data[col].dtype == "object"]
num_cols = [col for col in home_data.columns if home_data[col].dtype in ['int64', 'float64']]
missing_val_cols = [col for col in home_data.columns if home_data[col].isnull().any()] 

selected_cols = obj_cols + num_cols
X = home_data[selected_cols].copy()

# SPLIT DATA INTO TRAINING AND VALIDAION SETS #
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)


# # # # # # # EXPLORING TRAINING DATA # # # # # # #

print("Training Data | #rows: {}, #features: {} (#numerical: {}, #categorical: {}), #features with missing values: {}\n".format(X_train.shape[0], X_train.shape[1], len(num_cols), len(obj_cols), len(missing_val_cols)))


# # # # # # # DEFINE PREPROSSING STEPS # # # # # # #

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(
    steps=[('imputer', SimpleImputer(strategy='most_frequent')), 
           ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[('num', numerical_transformer, num_cols), 
                  ('cat', categorical_transformer, obj_cols)])


# # # # # # # DEFINE MODEL # # # # # # #

model = RandomForestRegressor(n_estimators=100, random_state=0)


# # # # # # # CREATE AND EVALUATE THE PIPELINE # # # # # # #

# CREATE
# Bundle preprocessing and modeling code in a pipeline
pipeline = Pipeline( 
    steps=[('preprocessor', preprocessor), 
           ('model', model)]) 
# FIT
pipeline.fit(X_train, y_train) # preprocessing of training data, fit model 
# PREDICT 
val_predictions = pipeline.predict(X_valid) # preprocessing of validation data, get predictions
# EVALUATE
val_mae = mean_absolute_error(y_valid, val_predictions)

print("Validation MAE | Pipeline -> {:,.0f}".format(val_mae))


# # Application 5 | Cross-Validation

# In **cross-validation**, we run our modeling process on different subsets of the data to get multiple measures of model quality.
# 
# For example, we could begin by dividing the data into 5 pieces, each 20% of the full dataset. In this case, we say that we have broken the data into 5 "**folds**". Then, we run one experiment for each fold:
# - In **Experiment 1**, we use the first fold as a validation (or **holdout**) set and everything else as training data. This gives us a measure of model quality based on a 20% holdout set.
# - In **Experiment 2**, we hold out data from the second fold (and use everything except the second fold for training the model). The holdout set is then used to get a second estimate of model quality.
# - We **repeat** this process, using every fold once as the holdout set. Putting this together, 100% of the data is used as holdout at some point, and we end up with a measure of model quality that is based on all of the rows in the dataset.
# 
# ### When cross-validation should be used?
# Cross-validation gives a **more accurate measure of model quality**, which is especially important if you are making a lot of modeling decisions. However, it **can take longer to run**, because it estimates multiple models (one for each fold).
# 
# So, given these tradeoffs, when should you use each approach?
# 
# - For small datasets, where extra computational burden isn't a big deal, you should run cross-validation.
# - For larger datasets, a single validation set is sufficient. Your code will run faster, and you may have enough data that there's little need to re-use some of it for holdout.
# 
# There's no simple threshold for what constitutes a large vs. small dataset. But if your model takes a couple minutes or less to run, it's probably worth switching to cross-validation.
# 
# Using cross-validation yields a much better measure of model quality, with the added benefit of cleaning up our code: note that we no longer need to keep track of separate training and validation sets. So, especially for small datasets, it's a good improvement!

# In[ ]:


# # # # # # # IMPORTS # # # # # # #

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


# # # # # # # GETTING DATA # # # # # # #

home_data = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')


# # # # # # # DATA MANIPULATION # # # # # # #

home_data.dropna(axis=0, subset=['SalePrice'], inplace=True) # remove rows with missing target from the dataset

# SELECT TARGET #
y = home_data.SalePrice

# SELECT FEATURES #
home_data.drop(['SalePrice'], axis=1, inplace=True) # drop the target column from the dataset
X = home_data.select_dtypes(exclude=['object']) # to keep things simple, we'll use only numerical predictors


# # # # # # # EXPLORING TRAINING DATA # # # # # # #

obj_cols = [col for col in X.columns if X[col].dtype == 'object'] 
num_cols = [col for col in X.columns if X[col].dtype != 'object'] 
missing_val_cols = [col for col in X.columns if X[col].isnull().any()] 

print("Training Data | #rows: {}, #features: {} (#numerical: {}, #categorical: {}), #features with missing values: {}\n".format(X.shape[0], X.shape[1], len(num_cols), len(obj_cols), len(missing_val_cols)))


# # # # # # # PREPROSESSING & MODELING # # # # # # #

pipeline = Pipeline(
    steps=[('preprocessor', SimpleImputer()), 
           ('model', RandomForestRegressor(n_estimators=50, random_state=0))])


# # # # # # # CROSS-VALIDATION # # # # # # #

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(pipeline, X, y, cv=3, scoring='neg_mean_absolute_error') # cv -> set the number of folds; scoring -> choose a measure of model quality to report

print("Validation MAE | Cross-Validation scores over 3 folds -> {}, average score -> {:,.0f}".format(scores, scores.mean()))


# Define a function to calculate MAE scores from different values for n_estimators
def get_score(n_estimators):
    """Return the average MAE over 3 CV folds of random forest model.
    
    Keyword argument:
    n_estimators -> the number of trees in the forest
    """
    pipeline = Pipeline(
        steps=[('preprocessor', SimpleImputer()), 
               ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=0))])
    scores = -1 * cross_val_score(pipeline, X, y, cv=3, scoring='neg_mean_absolute_error')
    return scores.mean()

candidate_n_estimators = [50, 100, 150, 200, 250, 300]
results = {est: get_score(est) for est in candidate_n_estimators} 
best_tree_number = min(results, key=results.get)

print("Validation MAE | Cross-Validation average score over 3 folds when best value of n_estimators ({}) is chosen -> {:,.0f}".format(best_tree_number, results[best_tree_number]))


# # Application 6 | Gradient Boosting

# **Gradient boosting** is a method that goes through cycles to iteratively add models into an ensemble.
# 
# It begins by initializing the ensemble with a single model, whose predictions can be pretty naive. (Even if its predictions are wildly inaccurate, subsequent additions to the ensemble will address those errors.)
# 
# Then, we start the cycle:
# 1. First, we use the current ensemble to **generate predictions** for each observation in the dataset. To make a prediction, we add the predictions from all models in the ensemble.
# 2. These predictions are used to **calculate a loss** function (like mean squared error, for instance).
# 3. Then, we use the loss function to **fit a new model** that will be added to the ensemble. Specifically, we determine model parameters so that adding this new model to the ensemble will reduce the loss. (Side note: The "gradient" in "gradient boosting" refers to the fact that we'll use *gradient descent* on the loss function to determine the parameters in this new model.)
# 4. Finally, we **add the new model to ensemble**, and ... repeat!
# 
# ## XGBoost
# **Extreme gradient boosting**, is an implementation of gradient boosting with several additional features focused on performance and speed.
# XGBoost has a few parameters that can dramatically affect accuracy and training speed:
# 
# ### n_estimators
# Specifies how many times to go through the modeling cycle. It is equal to the number of models that we include in the ensemble.
# > - Too low a value causes underfitting, which leads to inaccurate predictions on both training data and test data.
# > - Too high a value causes overfitting, which causes accurate predictions on training data, but inaccurate predictions on test data (which is what we care about).
# > - Typical values range from 100-1000, though this depends a lot on the *learning_rate* parameter.
# 
# ### early_stopping_rounds
# Offers a way to automatically find the ideal value for *n_estimators*. Early stopping causes the model to stop iterating when the validation score stops improving, even if we aren't at the hard stop for *n_estimators*. 
# > - It's smart to set a high value for *n_estimators* and then use early_stopping_rounds to find the optimal time to stop iterating.
# > - Since random chance sometimes causes a single round where validation scores don't improve, you need to specify a number for how many rounds of straight deterioration to allow before stopping. 
# > - *early_stopping_rounds = 5* is a reasonable choice.
# 
# ### eval_set
# Should be specified when *early_stopping_rounds* is used to set aside some data for calculating the validation scores.
# 
# ### learning_rate
# Instead of getting predictions by simply adding up the predictions from each component model, we can multiply the predictions from each model by a small number (known as the learning rate) before adding them in. 
# > - **This means each tree we add to the ensemble helps us less**. So, we can set a higher value for *n_estimators* without overfitting. If we use early stopping, the appropriate number of trees will be determined automatically.
# > - In general, a small learning rate and large number of estimators will yield more accurate XGBoost models, though it will also take the model longer to train since it does more iterations through the cycle. 
# > - As default, XGBoost sets learning_rate=0.1.
# 
# ### n_jobs
# On larger datasets where runtime is a consideration, you can use parallelism to build your models faster. It's common to set the parameter *n_jobs* equal to the number of cores on your machine. On smaller datasets, this won't help.

# In[ ]:


# # # # # # # IMPORTS # # # # # # #

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


# # # # # # # GETTING DATA # # # # # # #

home_data = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')


# # # # # # # DATA MANIPULATION # # # # # # #

home_data.dropna(axis=0, subset=['SalePrice'], inplace=True) # remove rows with missing target from the dataset

# SELECT TARGET #
y = home_data.SalePrice

# SELECT FEATURES #
home_data.drop(['SalePrice'], axis=1, inplace=True) # drop the target column from the dataset

missing_val_cols = [col for col in home_data.columns if home_data[col].isnull().any()]
home_data.drop(missing_val_cols, axis=1, inplace=True) # to keep things simple, we'll drop columns with missing values

obj_cols = [col for col in home_data.columns if home_data[col].nunique() < 10 and home_data[col].dtype == "object"]
num_cols = [col for col in home_data.columns if home_data[col].dtype in ['int64', 'float64']]
missing_val_cols = [col for col in home_data.columns if home_data[col].isnull().any()]

selected_cols = obj_cols + num_cols
X = home_data[selected_cols].copy()

# SPLIT DATA INTO TRAINING AND VALIDAION SETS #
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)


# # # # # # # EXPLORING TRAINING DATA # # # # # # #

print("Training Data | #rows: {}, #features: {} (#numerical: {}, #categorical: {}), #features with missing values: {}\n".format(X_train.shape[0], X_train.shape[1], len(num_cols), len(obj_cols), len(missing_val_cols)))


# # # # # # # ONE-HOT ENCODING # # # # # # #

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False) # handle_unknown='ignore' -> to avoid errors when the validation data contains classes that aren't represented in the training data; sparse=False -> ensures that the encoded columns are returned as a numpy array (instead of a sparse matrix)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[obj_cols])) 
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[obj_cols])) 

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (as they will be replaced with one-hot encoding)
num_X_train = X_train.drop(obj_cols, axis=1)
num_X_valid = X_valid.drop(obj_cols, axis=1)

# Add the one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)


# # # # # # # XGBOOST MODEL # # # # # # #

# DEFINE
xgboost_model = XGBRegressor(n_estimators=300, learning_rate=0.2) 
# FIT
xgboost_model.fit(OH_X_train, y_train, early_stopping_rounds=5, eval_set=[(OH_X_valid, y_valid)], verbose=False)
# PREDICT
val_predictions = xgboost_model.predict(OH_X_valid) 
# EVALUATE
val_mae = mean_absolute_error(val_predictions, y_valid) 

print("Validation MAE | XGBoost Model -> {:,.0f}".format(val_mae) + "\n")


model_1 = XGBRegressor(n_estimators=50, learning_rate=0.1)
model_2 = XGBRegressor(n_estimators=100, learning_rate=0.1)
model_3 = XGBRegressor(n_estimators=150, learning_rate=0.1)
model_4 = XGBRegressor(n_estimators=200, learning_rate=0.05)
model_5 = XGBRegressor(n_estimators=200, learning_rate=0.1)
model_6 = XGBRegressor(n_estimators=200, learning_rate=0.2)
model_7 = XGBRegressor(n_estimators=300, learning_rate=0.1)
model_8 = XGBRegressor(n_estimators=350, learning_rate=0.1)

models = [model_1, model_2, model_3, model_4, model_5, model_6, model_7, model_8]

# Function for comparing different models
def score_model(model, X_t=OH_X_train, X_v=OH_X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

for i in range(0, len(models)):
    val_mae = score_model(models[i])
    print("Validation MAE | XGBoost Model {} -> {:,.0f}".format(i+1, val_mae))

