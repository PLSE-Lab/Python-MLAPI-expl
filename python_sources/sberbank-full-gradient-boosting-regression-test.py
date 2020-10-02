#!/usr/bin/env python
# coding: utf-8

# # This notebook will contain a quick walk-through, going from original data to outcome.
# ## It contains limited EDA, which is contained in other notebooks.
# 
# I hope you will find some useful information in the notebook. Fork away if you find it useful. Please also leave comments if you have suggestions of what can be improved or what you'd like to see added.
# 
# ### The original data set is broken up into a number of testing data sets, namely
# 
#  1. Original: The original data set, where NaN data will be imputed
#     (30,500 rows, 292 features)
#  2. NA_Row: A data set with all rows containing NaN data removed (6,000
#     rows, 292 features)
#  3. NA_Col: A data set with all features containing NaN data removed
#     (30,500 rows, 241 features)
#  4. Clean: A data set with features containing > 500 NaNs removed,
#     followed by removal of those rows that contain NaNs (30,000 rows,
#     251 features)
#  5. Dummies: The clean data set, with dummy variables created for the
#     object features (30,000 rows, 391 features)
# 
# ### As a first algorithm, Gradient Boosting Regression is used to test on these data sets. I will be adding additional algorithms as time goes on.

# # Importing the main packages as well as training sets.
# I have done a minimal bit of pre-processing on some of the columns.

# In[ ]:


# Importing main packages and settings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[1]:


from keras.layers import Dense


# In[ ]:


# Loading the training dataset
df = pd.read_csv('../input/train.csv', parse_dates=['timestamp'])

# Adding feature for yearmonth of purchase and removing ID and timestamp
df['yearmonth'] = df['timestamp'].dt.year*100 + df["timestamp"].dt.month
df = df.drop(['id','timestamp'], axis=1)

# Adding log price for use as target variable
df['log_price_doc'] = np.log1p(df['price_doc'].values)


# # Creation of testing data sets

# In[ ]:


# df with all rows containing NA removed
df_na_row = df.dropna(axis=0)

# df with all features containing NA removed
df_na_col = df.dropna(axis=1)

# First 3 dataframes
print("Original DataFrame: {}".format(df.shape))
print("DataFrame After Dropping All Rows with Missing Values: {}".format(df_na_row.shape))
print("DataFrame After Dropping All Features with Missing Values: {}".format(df_na_col.shape))


# In[ ]:


# courtesy of SRK's notebook - to determine features with missing data
missing_df = df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.ix[missing_df['missing_count']>0]
ind = np.arange(missing_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(8,18))
rects = ax.barh(ind, missing_df.missing_count.values)
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
plt.show()


# ## Creation of a data set that only removes features with lots of missing data and with added dummies variables.
# Dropping only features with >500 NaNs and subsequently dropping rows with missing data results in preserving most of the rows and more features. Dummy variables are subsequently created for the object type features.

# In[ ]:


missing500_df = missing_df.ix[missing_df['missing_count']>500]
missing500_cols = missing500_df['column_name'].values
df_clean = df.drop(missing500_cols, axis=1).dropna(axis=0)
print("DataFrame After Dropping Features >500 NaN And Then Rows: {}".format(df_clean.shape))


# In[ ]:


df_dummies = pd.get_dummies(df_clean, drop_first=True)
print("Clean DataFrame With Dummy Variables: {}".format(df_dummies.shape))


# ## Creation of X features and y targets based on the data sets

# In[ ]:


# X and y based on original df, removing object type features
X_orig = df.drop(['price_doc', 'log_price_doc'], axis=1).select_dtypes(exclude=['object', 'datetime64']).values
y_orig = df['price_doc'].values

# X and y based on df with rows containing NaN dropped, removing object type features
X_row = df_na_row.drop(['price_doc', 'log_price_doc'], axis=1).select_dtypes(exclude=['object', 'datetime64']).values
y_row = df_na_row['price_doc'].values

# X and y based on df with features containing NaN dropped, removing object type features
X_col = df_na_col.drop(['price_doc', 'log_price_doc'], axis=1).select_dtypes(exclude=['object', 'datetime64']).values
y_col = df_na_col['price_doc'].values

# X and y based on df with dummy variables for object features
X_dummies = df_dummies.drop(['price_doc', 'log_price_doc'], axis=1).values
y_dummies = df_dummies['price_doc'].values


# In[ ]:


# Import the relevant sklearn packages
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor


# ## Initial test using Gradient Boosting Regressor

# In[ ]:


# removing warning just for now - will need to look into this
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# instantiating
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
scaler = StandardScaler()
gbr = GradientBoostingRegressor()

# setting up steps for the pipeline, with and without imputating
steps_exclimp = [('scaler', scaler),
        ('GradientBoostingRegressor', gbr)]

steps_inclimp = [('imputation', imp),
        ('scaler', scaler),
        ('GradientBoostingRegressor', gbr)]

# instantiating the pipeline
pipe = Pipeline(steps_exclimp)
pipe_imp = Pipeline(steps_inclimp)

# creating train ang test sets using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_dummies, y_dummies, test_size=0.3, random_state=42)

# fitting and predicting
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(pipe.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))


# ## Cross Validation Scores for the different data sets using Gradient Boosting Regressor

# In[ ]:


# Compute 3-fold cross-validation scores: cv_scores
cv_scores_dummies = cross_val_score(pipe, X_dummies, y_dummies, cv=3)

# Print the 3-fold cross-validation scores
print(cv_scores_dummies)

print("Average 3-Fold CV Score: {}".format(np.mean(cv_scores_dummies)))


# In[ ]:


# Compute 3-fold cross-validation scores: cv_scores
cv_scores_na_row = cross_val_score(pipe, X_row, y_row, cv=3)

# Print the 3-fold cross-validation scores
print(cv_scores_na_row)

print("Average 3-Fold CV Score: {}".format(np.mean(cv_scores_na_row)))


# In[ ]:


# Compute 3-fold cross-validation scores: cv_scores
cv_scores_na_col = cross_val_score(pipe, X_col, y_col, cv=3)

# Print the 3-fold cross-validation scores
print(cv_scores_na_col)

print("Average 3-Fold CV Score: {}".format(np.mean(cv_scores_na_col)))


# In[ ]:


# Compute 3-fold cross-validation scores: cv_scores
cv_scores_orig = cross_val_score(pipe_imp, X_orig, y_orig, cv=3)

# Print the 3-fold cross-validation scores
print(cv_scores_orig)

print("Average 3-Fold CV Score: {}".format(np.mean(cv_scores_orig)))


# In[ ]:




