#!/usr/bin/env python
# coding: utf-8

# # TPOT Automated ML Exploration with Ames Housing Regression
# ## By Jeff Hale
# 
# Let's see how TPOT does with a regression task with minimal data preparation. See my [Medium article on TPOT](https://medium.com/p/4c063b3e5de9/) for more information.

# ## Setup
# Let's import the libraries and methods we'll need and set some options to make data and charts display nicely.

# In[ ]:


# Import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import category_encoders as ce
import timeit
import category_encoders
import os
from math import sqrt
from sklearn.preprocessing import OneHotEncoder, Imputer, StandardScaler, MinMaxScaler, LabelEncoder, normalize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV,  cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics.scorer import make_scorer
from sklearn_pandas import CategoricalImputer
from tpot import TPOTRegressor

get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 300)


# In[ ]:


df = pd.read_csv("../input/train.csv")
print(df.head())


# In[ ]:


print(df.info())
print(df.describe())


# In[ ]:


# break into X and y dataframes
X = df.reindex(columns=[x for x in df.columns.values if x != 'SalePrice'])        # separate out X
y = df.reindex(columns=['SalePrice'])   # separate out y
y = np.ravel(y)                     # flatten the y array

# make list of numeric and string columns
numeric_cols = [] # could still have ordinal data
string_cols = []  # could have ordinal or nominal data

for col in X.columns:
    if (X.dtypes[col] == np.int64 or X.dtypes[col] == np.int32 or X.dtypes[col] == np.float64):
        numeric_cols.append(col)      # True integer or float columns
    
    if (X.dtypes[col] == np.object):  # Nominal and ordinal columns
        string_cols.append(col)


# In[ ]:


print(X[string_cols].head(2))


# In[ ]:


# impute missing values for string columns using sklearnpandas CategoricalImputer for string data
# s_imputer = CategoricalImputer(strategy="fixed_value", replacement="missing") 
# use above line as soon as sklearn-pandas updated
# s_imputer = CategoricalImputer()
X_string = X[string_cols]
# print(type(X_string))
# X_string = (s_imputer.fit_transform(X_string)

# or X_string = X_string.apply(s_imputer.fit_transform)

# X_string = pd.DataFrame(X_string, columns = string_cols)
X_string = X_string.fillna("missing")


# In[ ]:


# encode the X columns string values as integers
X_string = X_string.apply(LabelEncoder().fit_transform)  


# In[ ]:


print(X.head(2))


# In[ ]:


# imputing missing values with most freqent values for numeric columns
n_imputer = Imputer(missing_values='NaN', copy = True, strategy = 'most_frequent') # imputing with most frequent because some of these numeric columns are ordinal

X_numeric = X[numeric_cols]
X_numeric = n_imputer.fit_transform(X_numeric)
X_numeric = pd.DataFrame(X_numeric, columns = numeric_cols)


# In[ ]:


# add the string and numeric dataframes back together
X = pd.concat([X_numeric, X_string], axis=1, join_axes=[X_numeric.index])


# In[ ]:


X.info()


# In[ ]:


# convert to numpy array so that if gets XGBoost algorithm doesn't throw 
# ValueError: feature_names mismatch: ...
# see https://github.com/dmlc/xgboost/issues/2334
X = X.values


# In[ ]:


# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 55)


# In[ ]:


# Make a custom metric function for TPOT
# Root mean squared logarithmic error is how Kaggle scores this task
# Can't use custom scorer with n_jobs > 1.  Known issue.

# def custom_rmsle(y_true, y_pred):
#     return np.sqrt(np.mean((np.log(1 + y_pred) - np.log(1 + y_true))**2))

# Make a custom scorer from the custom metric function
# rmsle = make_scorer(custom_rmsle, greater_is_better=False)

# Number of pipelines is very small below so that we can quickly commit on Kaggle

# instantiate tpot 
tpot = TPOTRegressor(verbosity=3,  
                    random_state=55, 
                    #scoring=rmsle,
                    periodic_checkpoint_folder="intermediate_results",
                    n_jobs=-1, 
                    warm_start = True,
                    generations=20, 
                    population_size=80,
                    early_stop=8)
times = []
scores = []
winning_pipes = []

# run 2 iterations
for x in range(1):
    start_time = timeit.default_timer()
    tpot.fit(X_train, y_train)
    elapsed = timeit.default_timer() - start_time
    times.append(elapsed)
    winning_pipes.append(tpot.fitted_pipeline_)
    scores.append(tpot.score(X_test, y_test))
    tpot.export('tpot_ames.py')

# output results
times = [time/60 for time in times]
print('Times:', times)
print('Scores:', scores)   
print('Winning pipelines:', winning_pipes)

