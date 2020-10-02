#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn import preprocessing
import csv
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm
#import parameters_housing
from sklearn.preprocessing import LabelBinarizer
from catboost import CatBoostClassifier
from catboost import CatBoostRegressor

import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
print("Setup Complete")


# Note : Some of these imports were imported just for tuning/testing purpose and might be not used as such in this notebook.
# 
# Now let's load our train and test data. Please change path to file, if required.

# In[ ]:


train_data = pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv")
train_data.shape


# In[ ]:


test_data = pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv")
test_data.shape


# Let's print the list of columns in our training data

# In[ ]:


train_data.columns


# Let's separate our target variable 'SalePrice' from training data and store in a variable y

# In[ ]:


y = train_data['SalePrice']
y.shape


# Now, we need to drop 'SalePrice' column from training data

# In[ ]:


train_data.drop(['SalePrice'], axis=1,inplace=True)
train_data.shape


# As we can see from training data shape, the number of columns is reduced by 1.
# 
# Mostly, the models that we train do not work if any columns is having a missing value. So we need to clean our data first. The missing values will be filled based on their data type : numerical or categorical.
# 
# Let's define a function to get list of a specific category of columns.

# In[ ]:


def get_list_of_columns_of_type(X,col_type):
    s = (X.dtypes == 'object')     #default categorical
    if col_type == 'categorical':
        s = (X.dtypes == 'object')
    elif col_type == 'numerical':
        s = (X.dtypes != 'object')
    cols = list(s[s].index)
    return cols


# Now, let's test the above function and get lists of categorical columns in our training data.

# In[ ]:


categorical_cols = get_list_of_columns_of_type(train_data, 'categorical')
print(len(categorical_cols))
categorical_cols


# Let's print list of numerical columns

# In[ ]:


numerical_cols = get_list_of_columns_of_type(train_data, 'numerical')
print(len(numerical_cols))
numerical_cols


# Let's first analyze missing value counts in numerical columns in training data

# In[ ]:


missing_val_count_num_train = (train_data[numerical_cols].isnull().sum())
missing_val_count_num_train.sort_values(ascending=False,inplace=True)
missing_val_count_num_train[missing_val_count_num_train > 0]


# Let's look into missing value counts in categorical columns in training data

# In[ ]:


missing_val_count_cat_train = (train_data[categorical_cols].isnull().sum())
missing_val_count_cat_train.sort_values(ascending=False,inplace=True)
missing_val_count_cat_train[missing_val_count_cat_train > 0]


# Let us see how's the scenario of missing value counts of numerical data in test data

# In[ ]:


missing_val_count_num_test = (test_data[numerical_cols].isnull().sum())
missing_val_count_num_test.sort_values(ascending=False,inplace=True)
missing_val_count_num_test[missing_val_count_num_test > 0]


# In[ ]:


missing_val_count_cat_test = (test_data[categorical_cols].isnull().sum())
missing_val_count_cat_test.sort_values(ascending=False,inplace=True)
missing_val_count_cat_test[missing_val_count_cat_test > 0]


# As we can see the columns having missing values in test data is not same as columns having missing values in train data. It generally happens in real-time data, where we never know which all columns will be having missing values in future.
# 
# For this particular problem, as our train and test data is fixed. We can do this:-
# 1. Merge train and test data 
# 2. Fill missing in the merged set
# 3. Again de-merge into train and test data 

# In[ ]:


train_data_count = train_data.shape[0]
merged_data = train_data.append(test_data)
merged_data.shape


# Now let's print missing value counts for numerical and categorical columns for merged data

# In[ ]:


missing_val_count_num_merged = (merged_data[numerical_cols].isnull().sum())
missing_val_count_num_merged.sort_values(ascending=False,inplace=True)
missing_val_count_num_merged[missing_val_count_num_merged > 0]


# In[ ]:


missing_val_count_cat_merged = (merged_data[categorical_cols].isnull().sum())
missing_val_count_cat_merged.sort_values(ascending=False,inplace=True)
missing_val_count_cat_merged[missing_val_count_cat_merged > 0]


# As we can see, there are some columns having too many missing values. These columns are not might not be providing much information to predict our target variable. 
# We can remove those columns from our data by setting some threshold of say 50%.
# 
# Let's print the list of those columns

# In[ ]:


missing_val_count_cat_merged[missing_val_count_cat_merged > (merged_data.shape[0]/2)]


# Let's remove these columns from merged data

# In[ ]:


merged_data.drop(['PoolQC','MiscFeature','Alley','Fence'], axis=1,inplace=True)
merged_data.shape


# The list of numerical and categorical columns might have changed after removing those columns. Let's update them.

# In[ ]:


numerical_cols = get_list_of_columns_of_type(merged_data,'numerical')
categorical_cols = get_list_of_columns_of_type(merged_data,'categorical')
numerical_cols


# There are many ways to fill missing values in numerical columns:-
# 1. Fill with 0
# 2. Fill with mean
# 3. Fill with median
# 4. Fill with mode
# 
# For simplicity, let's fill the missing values by mean

# In[ ]:


for col in numerical_cols:
        merged_data[col] = merged_data[col].fillna(merged_data[col].mean())


# Let's check if there are any numerical columns left with missing values

# In[ ]:


missing_val_count_num_merged = (merged_data[numerical_cols].isnull().sum())
missing_val_count_num_merged.sort_values(ascending=False,inplace=True)
missing_val_count_num_merged[missing_val_count_num_merged > 0]


# As we can see the missing values in all the numerical columns have been filled.
# 
# Now, let's come to categorical columns.
# 
# For simplicity. we can fill missing values in categorical columns with constant "Unknown"

# In[ ]:


for col in categorical_cols:
        merged_data[col] = merged_data[col].fillna("Unknown")


# Let's check if there are any categorical columns left with missing values

# In[ ]:


missing_val_count_cat_merged = (merged_data[categorical_cols].isnull().sum())
missing_val_count_cat_merged.sort_values(ascending=False,inplace=True)
missing_val_count_cat_merged[missing_val_count_cat_merged > 0]


# As we can see the missing values in all the categorical columns have also been filled.
# 
# The models that we train do not understand categorical values in the form of strings as such. So, we need to encode them into 'int' or 'float'. 
# 
# Two most commonly used techniques are:-
# 1. Label Encoding : Assigns a number to each distinct value in a column starting from 0
# 2. One Hot Encoding : Replace that column with "number of unique values in that column" columns, where value will be 1 if that particular unique value was there in the original column in that row, and 0 otherwise
# 
# For simplicity, let's do Label Encoding

# In[ ]:


label_encoder = LabelEncoder()
for col in categorical_cols:
    merged_data[col] = label_encoder.fit_transform(merged_data[col])


# Let's check if there is any still any categorical column

# In[ ]:


categorical_cols = get_list_of_columns_of_type(merged_data,'categorical')
categorical_cols


# As we can see the list is empty. It means all columns have been converted into numerical columns.
# 
# Now we can feed this data to our model. But before that let's de-merge our training and test data

# In[ ]:


train_data = merged_data[0:train_data_count]
train_data.shape


# In[ ]:


test_data = merged_data[train_data_count:]
test_data.shape


# Now, we can feed this data to our model. But before that let's split our training data into training and validation data. So that we can test our model on unseen data for any biased-ness.

# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(train_data, y, train_size=0.8, test_size=0.2,random_state=0)
print(" Training Data Shape : {}\n Training Target Shape : {}\n Validation Data Shape : {}\n Validation Target Shape : {}".format(X_train.shape,y_train.shape,X_valid.shape,y_valid.shape))


# Let's use a GradientBoostRegresssor model with parameters(already tuned to best values) : n_estimators=4000, learning_rate=0.05, max_depth=4

# In[ ]:


my_model = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42) 


# Let's fit this model ONLY on X_train and y_train, as of now

# In[ ]:


my_model.fit(X_train, y_train)


# Let's check how our model behaves on the training data itself

# In[ ]:


preds = my_model.predict(X_train)
mean_absolute_error(y_train, preds)


# As we can see, this value is too low as expected. Model has predicted already seen data quite correctly. 
# 
# Let's test in on validation data which is unseen by the model

# In[ ]:


preds = my_model.predict(X_valid)
mean_absolute_error(y_valid, preds)


# As we can see there is large difference between Training Error and Validation Error. It means there is a certain biased-ness of our model towards training data.
# 
# Anyways, let's move on to make predictions for test data for submission. 
# 
# But before that we can make our model a little better by training it again on the entire training set which was there with us before train and valid split. So that we can get better results for our test data.

# In[ ]:


my_model.fit(train_data,y)


# In[ ]:


preds = my_model.predict(test_data)
preds.shape


# Let's load our submission format and fill 'SalePrice' columns with our predictions data

# In[ ]:


submission = pd.read_csv("/kaggle/input/home-data-for-ml-course/sample_submission.csv")
submission['SalePrice'] = preds


# Let's print submission to a .csv file for submission

# In[ ]:


submission.to_csv("submission.csv", index=False)

