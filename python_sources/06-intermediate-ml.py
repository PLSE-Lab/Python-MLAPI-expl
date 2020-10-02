#!/usr/bin/env python
# coding: utf-8

# [**Intermediate Machine Learning Micro-Course Home Page**](https://www.kaggle.com/learn/intermediate-machine-learning)
# 
# [Open on kaggle](https://www.kaggle.com/mahendrabishnoi2/06-intermediate-ml/edit)
# 
# ---

# # Dealing with missing values
# 
# **1) Drop Columns with missing values**
# <img src="https://i.imgur.com/Sax80za.png"/>

# **2) Imputation**: Filling missing values with some number. eg: mean, mode etc.
# <img src="https://i.imgur.com/4BpnlPA.png"/>

# **3) An Extension to Imputation**: Imputation, in addition with a new column showing location of imputed values.
# <img src="https://i.imgur.com/UWOyg4a.png"/>

# In[ ]:


# Data Loading
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

# Select target
y = data.Price

# To keep things simple, we'll use only numerical predictors
melb_predictors = data.drop(['Price'], axis=1)
X = melb_predictors.select_dtypes(exclude=['object'])

# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)


# In[ ]:


# Define function to measure quality of approach
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


# In[ ]:


# Approach 1: Drop columns with missing values

# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

# Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))


# In[ ]:


# Approach 2: Imputation

from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE from Approach 2 (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))


# In[ ]:


# Approach 3: An extension to Imputation

# Make copy to avoid changing original data (when imputing)
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

print("MAE from Approach 3 (An Extension to Imputation):")
print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))


# In[ ]:


# Shape of training data (num_rows, num_columns)
print(X_train.shape)

# Number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])


# Exercises of this tutorial are solved [here](https://www.kaggle.com/mahendrabishnoi2/exercise-missing-values)
# 
# ---

# # Categorical Variables
# 
# We can't use categorical values directly in our model. First we need to preprocess the data. 
# 
# **Threee Approaches:**
# 
# 1) Drop Categorical Variables: The easiest approach to dealing with categorical variables is to simply remove them from the dataset. This approach will only work well if the columns did not contain useful information.
# 
# 2) Label Encoding: Label encoding assigns each unique value to a different integer.
# <img src="https://i.imgur.com/tEogUAr.png"/>
# 
# 3) One-Hot Encoding: 
# <img src="https://i.imgur.com/TW5m0aJ.png"/>

# # Example

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

# Separate target from predictors
y = data.Price
X = data.drop(['Price'], axis=1)

# Divide data into training and validation subsets
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

# Drop columns with missing values (simplest approach)
cols_with_missing = [col for col in X_train_full.columns if X_train_full[col].isnull().any()] 
X_train_full.drop(cols_with_missing, axis=1, inplace=True)
X_valid_full.drop(cols_with_missing, axis=1, inplace=True)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = low_cardinality_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()


# In[ ]:


X_train.head()


# In[ ]:


# Get list of categorical variables
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


# In[ ]:


drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

print("MAE from Approach 1 (Drop categorical variables):")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))


# In[ ]:


# Approach 2: Label Encoding
from sklearn.preprocessing import LabelEncoder

# Make copy to avoid changing original data 
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

# Apply label encoder to each column with categorical data
label_encoder = LabelEncoder()
for col in object_cols:
    label_X_train[col] = label_encoder.fit_transform(X_train[col])
    label_X_valid[col] = label_encoder.transform(X_valid[col])

print("MAE from Approach 2 (Label Encoding):") 
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))


# **Approach 3:**
# We use the `OneHotEncoder` class from scikit-learn to get one-hot encodings. There are a number of parameters that can be used to customize its behavior.
# 
# We set `handle_unknown='ignore'` to avoid errors when the validation data contains classes that aren't represented in the training data, and
# setting `sparse=False` ensures that the encoded columns are returned as a numpy array (instead of a sparse matrix).
# To use the encoder, we supply only the categorical columns that we want to be one-hot encoded. For instance, to encode the training data, we supply `X_train[object_cols].(object_cols` in the code cell below is a list of the column names with categorical data, and so `X_train[object_cols]` contains all of the categorical data in the training set.)

# In[ ]:


# Approach 3: One hot encoding
from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

print("MAE from Approach 3 (One-Hot Encoding):") 
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))


# Exercises of this tutorial are solved [here](https://www.kaggle.com/mahendrabishnoi2/exercise-categorical-variables)
# 
# ---

# # Pipelines
# 
# ## Introduction
# **Pipelines** are a simple way to keep your data preprocessing and modeling code organized. Specifically, a pipeline bundles preprocessing and modeling steps so you can use the whole bundle as if it were a single step.
# 
# Many data scientists hack together models without pipelines, but pipelines have some important benefits. Those include:
# 
# 1. Cleaner Code
# 2. Fewer Bugs
# 3. Easier to Productionize
# 4. More Options for Model Validation

# In[ ]:


import os
os.listdir('../input/')


# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split

path = "../input/melbourne-housing-snapshot/melb_data.csv"

data = pd.read_csv(path)

data.head()


# In[ ]:


y = data.Price
X = data.drop('Price', axis = 1)


# In[ ]:


X.head()


# In[ ]:


y.head()


# In[ ]:


X_train_full, X_val_full, y_train, y_val = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)


# In[ ]:


# list all categorical columns
cat_columns = [col for col in X_train_full.columns if X_train_full[col].dtype == "object"]
cat_columns


# In[ ]:


# list columns with low cardinality
low_car_columns = [col for col in X_train_full.columns if X_train_full[col].nunique() < 10]
low_car_columns


# In[ ]:


# select intersection of categorical and low cardinality columns 
# i.e. categorical columns with low cardinality
categorical_cols = [col for col in X_train_full.columns
                   if X_train_full[col].nunique() < 10 and
                   X_train_full[col].dtype == "object"]
categorical_cols


# In[ ]:


# list all dtypes in our dataset
# print(data.dtypes)
# datatypes = 'object', 'int64', 'float64'


# In[ ]:


# we have three data types in our dataset
# 1. object (categorical columns)
# 2. int64 (integers) (numerical type)
# 3. float64 (numerical type)
# So now we will select numerical columns
numerical_cols = [col for col in X_train_full.columns if X_train_full[col].dtype == "int64" or X_train_full[col].dtype == "float64"]
numerical_cols


# In[ ]:


# select columns from original dataset to work with
cols = categorical_cols + numerical_cols

X_train = X_train_full[cols]
X_val = X_val_full[cols]


# In[ ]:


X_train.head()


# Three steps of creating a pipeline:
# 
# **Step 1: Define Preprocessing steps**
# 
# Similar to how a pipeline bundles together preprocessing and modeling steps, we use the `ColumnTransformer` class to bundle together different preprocessing steps. The code below:
# 
# - imputes missing values in numerical data, and
# - imputes missing values and applies a one-hot encoding to categorical data.

# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# preprocess numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# preprocess categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


# **Step 2: Define the Model**

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=0)


# **Step 3: Create and Evaluate the Pipeline**
# Finally, we use the `Pipeline` class to define a pipeline that bundles the preprocessing and modeling steps. There are a few important things to notice:
# 
# - With the pipeline, we preprocess the training data and fit the model in a single line of code. (In contrast, without a pipeline, we have to do imputation, one-hot encoding, and model training in separate steps. This becomes especially messy if we have to deal with both numerical and categorical variables!)
# - With the pipeline, we supply the unprocessed features in `X_valid` to the `predict()` command, and the pipeline automatically preprocesses the features before generating predictions. (However, without a pipeline, we have to remember to preprocess the validation data before making predictions.)

# In[ ]:


from sklearn.metrics import mean_absolute_error

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                             ('model', model)
                             ])

my_pipeline.fit(X_train, y_train)

preds = my_pipeline.predict(X_val)

score = mean_absolute_error(preds, y_val)

print("MAE: ", score)


# Exercises of this tutorial are solved [here](https://www.kaggle.com/mahendrabishnoi2/exercise-pipelines)
# 
# ---

# # Cross Validation
# In **cross-validation**, we run our modeling process on different subsets of the data to get multiple measures of model quality.
# 
# For example, we could begin by dividing the data into 5 pieces, each 20% of the full dataset. In this case, we say that we have broken the data into 5 "**folds**".
# <img src="https://i.imgur.com/9k60cVA.png"/>
# 
# Then, we run one experiment for each fold:
# 
# - In **Experiment 1**, we use the first fold as a validation (or holdout) set and everything else as training data. This gives us a measure of model quality based on a 20% holdout set.
# - In **Experiment 2**, we hold out data from the second fold (and use everything except the second fold for training the model). The holdout set is then used to get a second estimate of model quality.
# - We repeat this process, using every fold once as the holdout set. Putting this together, 100% of the data is used as holdout at some point, and we end up with a measure of model quality that is based on all of the rows in the dataset (even if we don't use all rows simultaneously).

# **When to use Cross Validation?**
# 
# Cross-validation gives a more accurate measure of model quality, which is especially important if you are making a lot of modeling decisions. However, it can take longer to run, because it estimates multiple models (one for each fold).
# 
# So, given these tradeoffs, when should you use each approach?
# 
# - For **small datasets**, where extra computational burden isn't a big deal, you should run cross-validation.
# - For **larger datasets**, a single validation set is sufficient. Your code will run faster, and you may have enough data that there's little need to re-use some of it for holdout.

# In[ ]:


import pandas as pd

data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]
y = data.Price


# In[ ]:


# define a pipeline that uses imputer to fill missing values and 
# random forests model to make predictions

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(
    steps = [
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators=50, random_state=0))
    ])


# In[ ]:


# use 'cross_val_score' function to obtain cross validation score

from sklearn.model_selection import cross_val_score

# Multiply by -1 because sklearn calculates negative MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                             cv=5,
                             scoring="neg_mean_absolute_error")

print("MAE: ", scores)


# In[ ]:


# Uncomment to check scorers supported by sklearn
# import sklearn
# sorted(sklearn.metrics.SCORERS.keys())


# In[ ]:


print("Average MAE: ", scores.mean())


# Exercises of this tutorial are solved [here](https://www.kaggle.com/mahendrabishnoi2/exercise-cross-validation)
# 
# ---

# # XGBoost
# 
# ## Gradient Boosting
# **Gradient boosting** is a method that goes through cycles to iteratively add models into an ensemble.
# 
# It begins by initializing the ensemble with a single model, whose predictions can be pretty naive. (Even if its predictions are wildly inaccurate, subsequent additions to the ensemble will address those errors).
# 
# - First, we use the current ensemble to generate predictions for each observation in the dataset. To make a prediction, we add the predictions from all models in the ensemble.
# - These predictions are used to calculate a loss function (like mean squared error, for instance).
# - Then, we use the loss function to fit a new model that will be added to the ensemble. Specifically, we determine model parameters so that adding this new model to the ensemble will reduce the loss. (Side note: The "gradient" in "gradient boosting" refers to the fact that we'll use [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) on the loss function to determine the parameters in this new model.)
# - Finally, we add the new model to ensemble, and ...
# - ... repeat!
# 
# <img src="https://i.imgur.com/MvCGENh.png"/>

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price

# Separate data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y)


# In[ ]:


# using xgboost library instead of sklearn because this library has
# several additional features focused on speed and performance.
from xgboost import XGBRegressor

my_model = XGBRegressor()
my_model.fit(X_train, y_train)


# In[ ]:


# Evaluate the model by xgboost
from sklearn.metrics import mean_absolute_error

preds = my_model.predict(X_valid)
mean_absolute_error(y_valid, preds)


# ## Parameter Tuning 
# - `n_estimators` - specifies how many times to go through the modeling cycle described above. It is equal to the number of models that we include in the ensemble.
#     - Too low a value causes underfitting, which leads to inaccurate predictions on both training data and test data.
#     - Too high a value causes overfitting, which causes accurate predictions on training data, but inaccurate predictions on test data (which is what we care about).
# - Typical values range from 100-1000, though this depends a lot on the `learning_rate` parameter.

# In[ ]:


my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train)


# - `early_stopping_rounds` - offers a way to automatically find the ideal value for `n_estimators`. Early stopping causes the model to stop iterating when the validation score stops improving, even if we aren't at the hard stop for `n_estimators`. It's smart to set a high value for `n_estimators` and then use `early_stopping_rounds` to find the optimal time to stop iterating.
# - Since random chance sometimes causes a single round where validation scores don't improve, we need to specify a number for how many rounds of straight deterioration to allow before stopping. Setting `early_stopping_rounds=5` is a reasonable choice. In this case, we stop after 5 straight rounds of deteriorating validation scores.
# - When using `early_stopping_rounds`, we also need to set aside some data for calculating the validation scores - this is done by setting the `eval_set` parameter.

# In[ ]:


my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train,
            early_stopping_rounds=5,
            eval_set=[(X_valid, y_valid)],
            verbose=False)


# - learning_rate
# 
# Instead of getting predictions by simply adding up the predictions from each component model, we can multiply the predictions from each model by a small number (known as the learning rate) before adding them in.
# 
# This means each tree we add to the ensemble helps us less. So, we can set a higher value for n_estimators without overfitting. If we use early stopping, the appropriate number of trees will be determined automatically.
# 
# In general, a small learning rate and large number of estimators will yield more accurate XGBoost models, though it will also take the model longer to train since it does more iterations through the cycle. As default, XGBoost sets `learning_rate=0.1`.

# In[ ]:


my_model = XGBRegressor(n_estimators=500, learning_rate=0.5)

my_model.fit(X_train, y_train,
            early_stopping_rounds=5,
            eval_set=[(X_valid, y_valid)],
            verbose=False)


# - `n_jobs`
# 
# On larger datasets where runtime is a consideration, we can use parallelism to build your models faster. It's common to set the parameter `n_jobs` equal to the number of cores on our machine. On smaller datasets, this won't help.
# 
# The resulting model won't be any better, so micro-optimizing for fitting time is typically nothing but a distraction. But, it's useful in large datasets where we would otherwise spend a long time waiting during the `fit` command.

# In[ ]:


my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)


# Exercises of this tutorial are solved [here](https://www.kaggle.com/mahendrabishnoi2/exercise-xgboost)
# 
# ---

# # Data Leakage
# **Data leakage** (or **leakage**) happens when your training data contains information about the target, but similar data will not be available when the model is used for prediction. This leads to high performance on the training set (and possibly even the validation data), but the model will perform poorly in production.
# 
# In other words, leakage causes a model to look accurate until you start making decisions with the model, and then the model becomes very inaccurate.
# 
# There are two main types of leakage: **target leakage** and **train-test contamination**.
# 
# **Target Leakage**: Occurs when our predictors include data that will not be available at the time you make predictions. It is important to think about target leakage in terms of the timing or chronological order that data becomes available, not merely whether a feature helps make good predictions.
# 
# Imagine we want to predict who will get sick with pneumonia. The top few rows of our raw data look like this:
# <img src="https://imgur.com/BEf5da3.png"/>
# 
# People take antibiotic medicines after getting pneumonia in order to recover. The raw data shows a strong relationship between those columns, but `took_antibiotic_medicine` is frequently changed after the value for `got_pneumonia` is determined. This is target leakage.
# 
# The model would see that anyone who has a value of False for `took_antibiotic_medicine` didn't have pneumonia. Since validation data comes from the same source as training data, the pattern will repeat itself in validation, and the model will have great validation (or cross-validation) scores.
# 
# But the model will be very inaccurate when subsequently deployed in the real world, because even patients who will get pneumonia won't have received antibiotics yet when we need to make predictions about their future health.
# 
# To prevent this type of data leakage, any variable updated (or created) after the target value is realized should be excluded.
# 
# <img src="https://i.imgur.com/y7hfTYe.png"/>

# **Train-Test Contamination**
# 
# A different type of leak occurs when we aren't careful to distinguish training data from validation data.
# 
# Recall that validation is meant to be a measure of how the model does on data that it hasn't considered before. We can corrupt this process in subtle ways if the validation data affects the preprocessing behavior. This is sometimes called train-test contamination.
# 
# For example, imagine we run preprocessing (like fitting an imputer for missing values) before calling `train_test_split()`. The end result? Our model may get good validation scores, giving us great confidence in it, but perform poorly when we deploy it to make decisions.
# 
# After all, we incorporated data from the validation or test data into how you make predictions, so the may do well on that particular data even if it can't generalize to new data. This problem becomes even more subtle (and more dangerous) when we do more complex feature engineering.
# 
# If our validation is based on a simple train-test split, exclude the validation data from any type of fitting, including the fitting of preprocessing steps. This is easier if we use scikit-learn pipelines. When using cross-validation, it's even more critical that you do your preprocessing inside the pipeline!

# In[ ]:


import pandas as pd

# Read the data
data = pd.read_csv('../input/aer-credit-card-data/AER_credit_card_data.csv', 
                   true_values = ['yes'], false_values = ['no'])

# Select target
y = data.card

# Select predictors
X = data.drop(['card'], axis=1)

print("Number of rows in the dataset:", X.shape[0])
X.head()


# In[ ]:


# small dataset so we will use cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

my_pipeline = make_pipeline(RandomForestClassifier(n_estimators=100))
scores = cross_val_score(my_pipeline, X, y,
                        cv=5,
                        scoring='accuracy')

print("Cross-validation accuracy: %f" % scores.mean())


# 98% accuracy doesn't happen often so enquire for target leakage.
# 
# Here is a summary of the data:
# 
# - **card**: 1 if credit card application accepted, 0 if not
# - **reports**: Number of major derogatory reports
# - **age**: Age n years plus twelfths of a year
# - **income**: Yearly income (divided by 10,000)
# - **share**: Ratio of monthly credit card expenditure to yearly income
# - **expenditure**: Average monthly credit card expenditure
# - **owner**: 1 if owns home, 0 if rents
# - **selfempl**: 1 if self-employed, 0 if not
# - **dependents**: 1 + number of dependents
# - **months**: Months living at current address
# - **majorcards**: Number of major credit cards held
# - **active**: Number of active credit accounts
# 
# A few variables look suspicious. For example, does `expenditure` mean expenditure on this card or on cards used before appying?
# 
# Here data comparision may be helpful

# In[ ]:


expenditures_cardholders = data.expenditure[y]
expenditures_noncardholders = data.expenditure[~y]

print('Fraction of those who did not receive a card and had no expenditures: %.2f'       %((expenditures_noncardholders == 0).mean()))
print('Fraction of those who received a card and had no expenditures: %.2f'       %(( expenditures_cardholders == 0).mean()))


# As shown above, everyone who did not receive a card had no expenditures, while only 2% of those who received a card had no expenditures. It's not surprising that our model appeared to have a high accuracy. But this also seems to be a case of target leakage, where expenditures probably means expenditures on the card they applied for.
# 
# Since `share` is partially determined by expenditure, it should be excluded too. The variables `active` and `majorcards` are a little less clear, but from the description, they sound concerning. In most situations, it's better to be safe than sorry if you can't track down the people who created the data to find out more.
# 
# We would run a model without target leakage as follows:

# In[ ]:


potential_leaks = ['expenditure', 'share', 'active', 'majorcards']
X = X.drop(potential_leaks, axis=1)

scores = cross_val_score(my_pipeline, X, y,
                        cv=5,
                        scoring="accuracy")

print(f"Cross-val accuracy: {scores.mean():.3f}" % scores.mean())


# This accuracy is quite a bit lower, which might be disappointing. However, we can expect it to be right about 80% of the time when used on new applications, whereas the leaky model would likely do much worse than that (in spite of its higher apparent score in cross-validation).
# 
# Data leakage can be multi-million dollar mistake in many data science applications. Careful separation of training and validation data can prevent train-test contamination, and pipelines can help implement this separation. Likewise, a combination of caution, common sense, and data exploration can help identify target leakage.
# 
# Exercises of this tutorial are solved [here](https://www.kaggle.com/mahendrabishnoi2/exercise-data-leakage)
# 
# ---
