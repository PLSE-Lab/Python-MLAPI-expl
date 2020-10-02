#!/usr/bin/env python
# coding: utf-8

# # General Approach
# This is a notebook for simple and efficient approach to generating a decent model.
# 
# Data Wrangling:
# My goal is to produce simple and reproducable code to wrangle the data for ensemble methods.
# 
# Feature Selection:
# Again, I will use `recursive feature elimination` to gain insights on the right amount of features.
# 
# Modeling:
# Simple ensemble modeling

# In[ ]:


import pandas as pd # data wrangling & cleaning & feature engineering
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # data visualization
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "BackendInline.figure_format = 'retina' # makes visualization retina display friendly")

# Machine Learning libraries for modeling
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# In[ ]:


# load datasets as pd.DataFrame
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# cache 'Id' column for submission file
testid = test['Id']

# drop irrelevant columns
test.drop(['Id'], axis = 1, inplace = True)
train.drop(['Id'], axis = 1, inplace = True)


# ## First look at the dataset.
# 
# Two main challenges are caregorical variables and null values.

# In[ ]:


train.head(7)


# In[ ]:


def print_null(df):   # Checking for null values
    '''
    print_null(dataframe)
    ...
    From a dataframe, prints:
    column name | number of null values
    '''
    for col, val in zip(df.columns, df.isnull().sum()):
        if val > 0:
            print("Column: {0:15} | Nulls: {1}".format(col, val))


# In[ ]:


print_null(train)


# ## Data Wrangling
# 
# Some features has a significant amount of null values and categorical variables as strings.
# 
# 1) Dealing with a massive amount of categorical variables.
# 
# a. To make sure we take into account all variables, I made a function `dummify` that takes in a `DataFrame` and returns a new `DataFrame` with all categorical & string variables into dummified columns.
# 
# b. Because of the vast number of features in the datasets, not all dummified features from `train` are present in the `test` dataset. The function `match_columns` takes in two `DataFrames` and only keeps colums where the column names are equal.
#  
# 
# 2) Dealing with a massive amount of `NaN`.
# 
# a. After filling 0's for null values, running `dummify` will take care of the categorical nulls. 
# 
# ## Example:
# 
# Before
# 
# | A |
# | ------- |
# | Good |
# | Poor | 
# | `NaN` | 
# 
# After  `dummify`
# 
# | A_Good | A_Poor |
# | ------- | ------- |
# | 1 | 0 |
# | 0 | 1 |
# | 0 | 0 |
# 

# In[ ]:


def dummify(df):
    '''
    dummify(DataFrame)
    ...
    Iterates through the dataframe and returns 
    binarized categorical features (strings).
    Ignores features with int values.
    Returns value is a pandas dataframe.
    '''
    df1 = pd.DataFrame()
    for i in df.columns:
        if np.dtype(df[i]) == 'O':
            df[i].fillna(0, inplace=True)
            for index in range(0, len(df[i].unique())):
                df1[str(i) + '_' + str(df[i].unique()[index])] =                pd.get_dummies(df[i]).iloc[:,[index]]
        else:
            df1[i] = df[i]
    return df1


# In[ ]:


train_dummified = dummify(train)
test_dummified = dummify(test)


# In[ ]:


# Distribution of my correlation with almost 300 features. 
# Over 100 features have near-0 correlation.
plt.figure(figsize=(12,10))
plt.hist(train_dummified.corr()['SalePrice'].values, bins=50)
plt.title('Correlation of SalePrice & other features')
plt.xlabel('pearson correlation')
plt.ylabel('count')


# Some engineered dummy features are not co-existent between two DataFrames.
# 
#  `match_columns` function will iterate through the two DataFrames and eliminate unnecessary features.

# In[ ]:


print(train_dummified.shape)
print(test_dummified.shape)


# In[ ]:


def match_columns(df1, df2):
    '''
    match_columns(DataFrame1, DataFrame2)
    ...
    keeps columns with matching names
    drops columns with names that do not exist in either DataFrame 
    returns DataFrame1, DataFrame2 with all matching column names
    '''
# iterate through the two columns, drop columns that only exist in one of the dataframes
    for col in df1.columns:
        if col not in df2.columns:
            df1.drop(col, axis = 1, inplace = True)
    for col in df2.columns:
        if col not in df1.columns:
            df2.drop(col, axis = 1, inplace = True)
            
# reorder one of the DataFrames to match the order of columns
    ordered_df2 = pd.DataFrame()
    for col in df1.columns: 
        ordered_df2[col] = df2[col]
        
    return df1, ordered_df2


# In[ ]:


df_train, df_test = match_columns(train_dummified, test_dummified)


# In[ ]:


# Checking for columns that do not match
for x,y in zip(df_train.columns, df_test.columns):
    if x != y:
        print(x, y)


# In[ ]:


print(df_train.shape)
print(df_test.shape)


# Now, the only null values left are `ints` and `floats`
# 
# I've decided to fill them with 0, since the null values probably mean non-existent.

# In[ ]:


# visualizing null values 
print_null(df_train)


# In[ ]:


print_null(df_test)


# In[ ]:


# Filling the rest of the null values
# Note: this will only work after running dummify function.
def fillna_median(df):
    '''
    fillna_median(dataframe)
    ...
    From a dataframe, fills null with median values
    '''
    for col, val in zip(df.columns, df.isnull().sum()):
        if val > 0:
            df[col].fillna(df[col].median(), inplace=True)
    return df


# In[ ]:


df_train.fillna(0, inplace=True)
df_test.fillna(0, inplace=True)


# In[ ]:


# Setting X and y values
y = train.SalePrice.values
X = df_train


# ### Feature Elimination with RFE
# 
# With 285 features, my model may be more optimized if I eliminate some features.
# Below function will return a grid of cross-validated scores of different number of features, and different estimators.
# Number of features and estimators are both customizable, but this function may take a long time depending on the number of features. For this kernel, I will not run this function.

# In[ ]:


def rfe_score_grid(X, y, recursion_denom = 10):
    '''
    rfe_score_grid(X, y, recursion_denom = 10)
    ...
    
    returns a dataframe with rfe values, different models,
    and its relative cross validation score.
    Recursion denom is the number that will divide by 
    
    '''
    grid = pd.DataFrame(columns=['RFE', 'GradientBoosting', 'RandomForest'])
    feature_num = [i for i in range(1, X.shape[1]) if i % recursion_denom == 0]
    gb = GradientBoostingRegressor(n_estimators=100, random_state=33)
    rf = RandomForestRegressor(n_estimators=100, random_state=33)
    gb_list = []
    rf_list = []
    for i in feature_num:
        rfe_gb = RFE(gb, n_features_to_select=i, step=recursion_denom)
        gb_list.append(cross_val_score(rfe_gb, X, y, cv = 3, n_jobs = 4).mean())
        rfe_rf = RFE(rf, n_features_to_select=i, step=recursion_denom)
        rf_list.append(cross_val_score(rfe_rf, X, y, cv = 3, n_jobs = 4).mean())
    
    grid['RFE'] = feature_num
    grid['GradientBoosting'] = gb_list
    grid['RandomForest'] = rf_list
    return grid


# In[ ]:


gb = GradientBoostingRegressor(n_estimators=100)
rf = RandomForestRegressor(n_estimators=200)

# Simple cross-validation
print('Gradient Boosting score =', cross_val_score(gb, X, y, cv = 4).mean())
print('Random Forest score =', cross_val_score(rf, X, y, cv = 4).mean())


# Looks like Gradient Boosting works much better.

# In[ ]:


gb = GradientBoostingRegressor()
rfe = RFE(gb, n_features_to_select=50)
rfe.fit(X, y)


# In[ ]:


X_rfe = pd.DataFrame()
for keep, col in zip(rfe.support_, X.columns):
    if keep == True:
        X_rfe[col] = X[col]
    else:
        continue

X_rfe, X_test_rfe = match_columns(X_rfe, df_test)


# In[ ]:


model = GradientBoostingRegressor(loss = 'ls', max_depth = 3, max_features= 'log2', n_estimators = 1500)

model.fit(X_rfe, y)


# In[ ]:


predict = model.predict(X_test_rfe)
submission = pd.DataFrame()
submission['Id'], submission['SalePrice'] =  testid, predict
submission.to_csv(path_or_buf = 'Submission.csv', index = False)


# I welcome any feedback.
# 
# # Thank you!
