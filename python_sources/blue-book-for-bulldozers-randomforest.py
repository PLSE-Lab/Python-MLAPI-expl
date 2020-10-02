#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## 1. Problem definition 
# > How well can we predict the future sale price of a bulldozer, given its characteristics and previous examples of how much similiar bulldozers have been sold for?

# ## 2. Data
# 
# The data is downloaded from the Kaggle Bluebook for Bulldozers competition: https://www.kaggle.com/c/bluebook-for-bulldozers/data
# 
# There are 3 main datasets:
# 
# * Train.csv is the training set, which contains data through the end of 2011.
# * Valid.csv is the validation set, which contains data from January 1, 2012 - April 30, 2012.
# * Test.csv is the test set, which won't be released until the last week of the competition. It contains data from May 1, 2012 - November 2012.

# ## 3. Evaluation
# 
# The evaluation metric for this competition is the RMSLE (root mean squared log error) between the actual and predicted auction prices.
# 
# For more on the evaluation of this project check: https://www.kaggle.com/c/bluebook-for-bulldozers/overview/evaluation
# 
# **Note:** The goal for most regression evaluation metrics is to minimize the error.

# ## 4. Features
# 
# Kaggle provides a data dictionary detailing all of the features of the dataset. You can view this data dictionary on Google Sheets: https://docs.google.com/spreadsheets/d/18ly-bLR8sbDJLITkWG7ozKm8l3RyieQ2Fpgix-beSYI/edit?usp=sharing

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn


# In[ ]:


# Import training and validation sets
df = pd.read_csv("../input/bluebook-for-bulldozers/TrainAndValid.csv",
                low_memory=False)


# In[ ]:


df.info()


# In[ ]:


df.isna().sum()


# In[ ]:


df.columns


# In[ ]:


fig, ax = plt.subplots()
ax.scatter(df["saledate"][:1000], df["SalePrice"][:1000]);


# In[ ]:


df.saledate[:1000]


# In[ ]:


df.saledate.dtype


# In[ ]:


df.SalePrice.plot.hist();


# # Parsing dates
# When we work with time series data, we want to enrich the time & date component as much as possible.
# 
# We can do that by telling pandas which of our columns has dates in it using the `parse_dates` parameter.

# In[ ]:


# Import data again but this time parse dates
df = pd.read_csv("../input/bluebook-for-bulldozers/TrainAndValid.csv",
                low_memory=False,
                parse_dates=["saledate"])


# In[ ]:


df.saledate.dtype


# In[ ]:


df.saledate[:1000]


# In[ ]:


fig, ax = plt.subplots()
ax.scatter(df["saledate"][:1000], df["SalePrice"][:1000]);


# In[ ]:


df.head()


# In[ ]:


df.head().T


# In[ ]:


df.saledate.head(20)


# # Sort DataFrame by saledate
# When working with time series data, it's a good idea to sort it by date.

# In[ ]:


# Sort DataFrame in date order
df.sort_values(by=["saledate"], inplace=True, ascending=True)
df.saledate.head(20)


# In[ ]:


# Make a copy of the original DataFrame to perform edits on
df_tmp = df.copy()


# # Add datetime parameters for `saledate` column

# In[ ]:


df_tmp["saleYear"] = df_tmp.saledate.dt.year
df_tmp["saleMonth"] = df_tmp.saledate.dt.month
df_tmp["saleDay"] = df_tmp.saledate.dt.day
df_tmp["saleDayOfWeek"] = df_tmp.saledate.dt.dayofweek
df_tmp["saleDayOfYear"] = df_tmp.saledate.dt.dayofyear


# In[ ]:


df_tmp.head().T


# In[ ]:


# Now we've enriched our DataFrame with date time features, we can remove 'saledate'
df_tmp.drop("saledate", axis=1, inplace=True)


# In[ ]:


# Check the values of different columns
df_tmp.state.value_counts()


# In[ ]:


df_tmp.head()


# In[ ]:


len(df_tmp)


# ## 5.Modelling
# Let's do some model-driven EDA.

# In[ ]:


# Building a machine learning model
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_jobs=-1,
                             random_state=12)

model.fit(df_tmp.drop("SalePrice", axis=1), df_tmp["SalePrice"])


# In[ ]:


df_tmp.info()


# In[ ]:


df_tmp["Thumb"].dtype


# In[ ]:


df_tmp.isna().sum()


# # Convert string to categories
# One way we can turn all our data into numbers is by converting them into pandas categories.

# In[ ]:


df_tmp.head().T


# In[ ]:


pd.api.types.is_string_dtype(df_tmp["Thumb"])


# In[ ]:


# Find the columns which contain string
for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        print(label)


# In[ ]:


# This will turn all of the string value into category values
for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        df_tmp[label] = content.astype("category").cat.as_ordered()


# In[ ]:


df_tmp.info()


# In[ ]:


df_tmp.state.cat.categories


# In[ ]:


df_tmp.state.cat.codes


# In[ ]:


# Check missing data
df_tmp.isnull().sum()/len(df_tmp)


# # Save preprocessed data

# In[ ]:


# Export current tmp dataframe
df_tmp.to_csv("/kaggle/working/train_tmp.csv",
             index=False)


# In[ ]:


# Import preprocessed data
df_tmp = pd.read_csv("/kaggle/working/train_tmp.csv",
                    low_memory=False)
df_tmp.head().T


# In[ ]:


df_tmp.isna().sum()


# # Fill missing values

# In[ ]:


# Fill numerical missing values first
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        print(label)


# In[ ]:


df_tmp.ModelID


# In[ ]:


# Check for which numeric columns have null values
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            # Add a binary column, tells if the data was missing or not
            df_tmp[label +"_is_missing"] = pd.isnull(content)
            # Fill missing numeric values with median
            df_tmp[label] = content.fillna(content.median())


# In[ ]:


# Check if there's any null numeric values
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)


# In[ ]:


# Check to see how many examples were missing
df_tmp.auctioneerID_is_missing.value_counts()


# In[ ]:


df_tmp.isna().sum()


# # Filling and turning categorical variables into numbers

# In[ ]:


# Check for columns which aren't numeric
for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        print(label)


# In[ ]:


# Turn categorical variables into numbers and fill missing
for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        # Add binary column to indicate whether sample had missing value
        df_tmp[label+"_is_missing"] = pd.isnull(content)
        # Turn categories into numbers and add +1
        df_tmp[label] = pd.Categorical(content).codes+1


# In[ ]:


pd.Categorical(df_tmp["state"]).codes+1


# In[ ]:


df_tmp.info()


# In[ ]:


df_tmp.head().T


# In[ ]:


df_tmp.isna().sum()


# Now that all of data is numeric as well as our dataframe has no missing values, we should be able to build a machine learning model.

# In[ ]:


df_tmp.head()


# In[ ]:


len(df_tmp)


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Instantiate model\nmodel = RandomForestRegressor(n_jobs=-1,\n                             random_state=12)\n\n# Fit the model\nmodel.fit(df_tmp.drop("SalePrice", axis=1), df_tmp["SalePrice"])')


# In[ ]:


# Score the model
model.score(df_tmp.drop("SalePrice", axis=1), df_tmp["SalePrice"])


# # Spliting data into train/validation sets

# In[ ]:


df_tmp.saleYear


# In[ ]:


df_tmp.saleYear.value_counts()


# In[ ]:


# Split data into training and validation
df_val = df_tmp[df_tmp.saleYear == 2012]
df_train = df_tmp[df_tmp.saleYear != 2012]

len(df_val), len(df_train)


# In[ ]:


# Split data into X & y
X_train, y_train = df_train.drop("SalePrice", axis=1), df_train.SalePrice
X_valid, y_valid = df_val.drop("SalePrice", axis=1), df_val.SalePrice

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape


# # Building an evaluation function

# In[ ]:


# Create evaluation  function (the competition uses RMSLE)
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score

def rmsle(y_test, y_preds):
    """
    Calculates root mean squared error between predictions and truelabels.
    """
    return np.sqrt(mean_squared_log_error(y_test, y_preds))

# Create function to evaluate model on a few different levels
def show_scores(model):
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_valid)
    scores = {"Training MAE": mean_absolute_error(y_train, train_preds),
             "Valid MAE": mean_absolute_error(y_valid, val_preds),
             "Training RMSLE": rmsle(y_train, train_preds),
             "Valid RMSLE": rmsle(y_valid, val_preds),
             "Training R^2": r2_score(y_train, train_preds),
             "Valid R^2": r2_score(y_valid, val_preds)}
    return scores


# In[ ]:


# It's takes too long for experiment
#%%time
#model = RandomForestRegressor(n_jobs=-1,
 #                            random_state=12)

#model.fit(X_train, y_train)


# In[ ]:


len(X_train), len(y_train)


# In[ ]:


# Change max_samples value
model = RandomForestRegressor(n_jobs=-1,
                             random_state=12,
                             max_samples=10000)


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Cutting down on the max number of samples each estimator can see improves training time\nmodel.fit(X_train, y_train)')


# In[ ]:


(X_train.shape[0] * 100 / 1000000)


# In[ ]:


10000 * 100


# In[ ]:


show_scores(model)


# # Hyperparameter tunning with RandomizedSearchCV

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.model_selection import RandomizedSearchCV\n\n# Different RandomForestRegressor hyperparameters\nrf_grid = {"n_estimators": np.arange(10, 100, 10),\n          "max_depth": [None, 3, 5, 10],\n          "min_samples_split": np.arange(2, 20, 2),\n          "min_samples_leaf": np.arange(1, 20, 2),\n          "max_features": [0,5, 1, "sqrt", "auto"],\n          "max_samples": [10000]}\n\n# Instantiate RandomizedSearchCV\nrs_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,\n                                                   random_state=12),\n                             param_distributions=rf_grid,\n                             n_iter=2,\n                             cv=5,\n                             verbose=True)\n\n# Fit the RandomizedSearchCV\nrs_model.fit(X_train, y_train)')


# In[ ]:


# Find the best model hyperparameters
rs_model.best_params_


# In[ ]:


# Evaluate the RandomizedSearch model
show_scores(rs_model)


# ### Train a model with the best hyperparameters
# **Note:** These were found after 100 iterations of `RandomizedSearchCV`.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Most ideal hyperparameters\nideal_model = RandomForestRegressor(n_estimators=40,\n                                   min_samples_leaf=1,\n                                   min_samples_split=14,\n                                   max_features=0.5,\n                                   n_jobs=-1,\n                                   max_samples=None,\n                                   random_state=12)\n\n# Fit the ideal model\nideal_model.fit(X_train, y_train)')


# In[ ]:


# Scores for ideal_model (trained on all the data)
show_scores(ideal_model)


# In[ ]:


# Scores on rs_model (only trained on ~10,000 examples)
show_scores(rs_model)


# # Make predictions on test data

# In[ ]:


# Import the test data
df_test = pd.read_csv("../input/bluebook-for-bulldozers/Test.csv",
                     low_memory=False,
                     parse_dates=["saledate"])

df_test.head()


# In[ ]:


# Make predictions on the test dataset
test_preds = ideal_model.predict(df_test)


# # Preprocessing the data (getting the test dataset in the same format as our training dataset)

# In[ ]:


def preprocess_data(df):
    """
    Performs transformations on df and returns transformed df.
    """
    df["saleYear"] = df.saledate.dt.year
    df["saleMonth"] = df.saledate.dt.month
    df["saleDay"] = df.saledate.dt.day
    df["saleDayOfWeek"] = df.saledate.dt.dayofweek
    df["saleDayOfYear"] = df.saledate.dt.dayofyear
    
    df.drop("saledate", axis=1, inplace=True)
    
    # Fill the numeric rows with median
    for label, content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum(): 
                # Add a binary column which tells us if the data was missing or not
                df[label+"_is_missing"] = pd.isnull(content)
                # Fill missing numeric values with median
                df[label] = content.fillna(content.median())
    
        # Filled categorical missing data and turn categories into numbers
        if not pd.api.types.is_numeric_dtype(content):
            df[label+"_is_missing"] = pd.isnull(content)
            # We add +1 to the category code because pandas encodes missing categories as -1
            df[label] = pd.Categorical(content).codes+1
    
    return df 


# In[ ]:


# Process the test data
df_test = preprocess_data(df_test)
df_test.head()


# In[ ]:


# Make predictions on updated test data
test_preds = ideal_model.predict(df_test)


# In[ ]:


X_train.head()


# In[ ]:


# We can find how the columns differ using sets
set(X_train.columns) - set(df_test.columns)


# In[ ]:


# Manually adjust df_test to have auctioneedID_is_missing column
df_test["auctioneedID_is_missing"] = False
df_test.head()


# Finally now our test dataframe has the same features as our training dataframe, we can make predictions!

# In[ ]:


# Make predictions on the test data
test_preds = ideal_model.predict(df_test)


# In[ ]:


test_preds


# Predictions are not in the same format Kaggle is asking for

# In[ ]:


df_preds = pd.DataFrame()
df_preds["SalesID"] = df_test["SalesID"]
df_preds["SalesPrice"] = test_preds
df_preds


# In[ ]:


# Export prediction data
df_preds.to_csv("/kaggle/working/test_predictions.csv", index=False)


# # Featute Importance
# Feature importance seeks to figure out which different attributes of the data were most importance when it comes to predicting the **target variable** (SalePrice).

# In[ ]:


# Find feature importance of our best model
ideal_model.feature_importances_


# In[ ]:


# Helper function for plotting importance
def plot_features(columns, importances, n=20):
    df = (pd.DataFrame({"features": columns,
                       "feature_importances": importances,})
         .sort_values("feature_importances", ascending=False)
         .reset_index(drop=True))
    
    # Plot the dataframe
    fig, ax = plt.subplots()
    ax.barh(df["features"][:n], df["feature_importances"][:20])
    ax.set_ylabel("Features")
    ax.set_xlabel("Feature importance")
    ax.invert_yaxis()


# In[ ]:


plot_features(X_train.columns, ideal_model.feature_importances_)


# In[ ]:


df["Enclosure"].value_counts()


# In[ ]:




