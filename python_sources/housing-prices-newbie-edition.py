#!/usr/bin/env python
# coding: utf-8

# # Introduction
# The goal of this notebook is to build a ML model that can accurately predict housing prices from Ames, Iowa based on the features available from the housing data set. Since we're predicing pricing we will most likely be using a regression model as our final model.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas_profiling

# Data Viz
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(rc={'figure.figsize':(16,16)})

# import the data
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
df_joined = pd.concat([df_train, df_test], sort=False)


# # Data Exploration
# 
# ## High Level
# Let's first see what we're working with. It's important for us to get a feel for the data so we can start anticipating the cleaning, pre-processing and potential feature engineering. We also want to start understanding feature relationships as well. 
# 
# Since we have so much data, relatively speaking, we're only interested in the top level aggregate numbers for now. The total number of features available make it too cumbersome to really dive deep on each column as opposed to a dataset like the Titanic dataset. As we can see we have only 3 different types of data (2 of which are numerical) a majority of the columns are *object* type which means we could potentially be dealing with a lot of categorical data. Almost half of the total columns have null values and 6 of those columns have over half null values. The prediction target here is the **SalePrice** column. 
# 
# ## Hypotheses
# Based on the features we have available to us paired with the contextual knowledge of housing prices and costs we can make some hypotheses about which factors would have a large impact on the sale price for a house. Generally speaking homes that have more in terms of size and amenities will be more expensive which in our case include: *Dwelling Type*, *Lot Area*, *House Style*, *Basement Size*, *First Floor Size*, *Second Floor Size*, *Wood Deck*, *Pool Area*, etc. 
# 
# Homes that are most recently built along with the top quality condition which include: *Pool Quality*, *Fence Quality*, *Garage Quality*, *Fireplace Quality*, *Kitchen Quality*, *Heating Quality*, *Basement Quality*, *Basement Condition*, *External Quality*, *Overall Condition*, *Overall Quality*, etc. Some interesting features that may also have a great impact include *SaleCondition* where the house maybe foreclosed so the price, we assume, would be low; *SaleType* where the home for example could've been sold as soon as it was done constructed.
# 
# The feature that can play a big factor would be the location of the house there are 26 different locations present and some of these locations may have a large disparity in pricing for example because maybe a university is present. If we wanted to dive deeper we can perhaps do some internet digging to find what the housing market looks like for these places using something like a zillow and align it the year that this dataset speaks for.   
# 

# In[ ]:


df_joined.head()


# In[ ]:


print("Total Columns: %s\nTotal Rows: %s" % (df_joined.shape[1], df_joined.shape[0]))


# In[ ]:


types = {}
for d in df_joined.dtypes.tolist():
    if d not in types.keys():
        types[d] = 1
    else:
        types[d] += 1
print("Total count of column types\n-------------------")
types


# In[ ]:


print("Total columns with null values: %s" % (len(df_joined.columns[df_joined.isna().any()]),))


# In[ ]:


print("Columns with null values\n-------------------")
null_series = pd.isnull(df_joined).sum()
null_series[null_series > 0]


# In[ ]:


types = {}
indices = null_series[null_series > 0].index.tolist()
for d in df_joined[indices].dtypes.tolist():
    if d not in types.keys():
        types[d] = 1
    else:
        types[d] += 1
print("Total count of column types for columns with null values\n-------------------")
types


# In[ ]:


# df_joined.profile_report(style={'full_width': True})


# # Data Visualization
# Now let's start poking at the hypotheses we made above by using some graphing to infer some relationships.
# 
# The newer the house the higher the price hypothesis seems to be correct as we can see below indicated by the **Sale Type**, the **Sale Condition** (the *Partial* type indicates that it is a new home see documentation) and the **MSSubClass** where the *1946 & Newer* model homes sold for more comparatively. 
# 
# The assumption about the higher quality and higher condition seems to also maintain. As we can see from the **Heating Quality**, **External Quality**, **Kitchen Quality**, **Basement Condition** and generally speaking for **External Condition**,**Overall Condition** and **Functional** type. The Garage quality and condition doesn't have a strong correlation between higher quality/condition with pricing. Maybe that's because people care less about what the garage looks like maybe.
# 
# Finally, the sizing hypothesis seems to stand correct as well if you look at the **First floor**, **Second floor**, **Basement** and **Masonry veneer type**. Although the **Lot area** doesn't have a strong impact.
# 
# There are several other features here that seem real interesting and impactful more importantly they look more like a blanket category. For example the **Year Sold** looks like it can be effective because we know in real estate the market can be up or down from period to period. The same goes for the **Neighborhood** as we know that pricing is consistent and relatively similarly priced for all houses in the neighborhood. **Building type** also makes sense because generally building types will be consistent in pricing across any attribute.  
# 
# 

# ## Newer Houses

# In[ ]:


sns.barplot(x='SaleType', y='SalePrice', data=df_train)


# In[ ]:


sns.pointplot(x='SaleCondition', y='SalePrice', data=df_train)


# In[ ]:


sns.pointplot(x='MSSubClass', y='SalePrice', data=df_train)


# ## Quality & Condition

# In[ ]:


sns.pointplot(x='GarageQual', y='SalePrice', data=df_train, order=['Po', 'Fa', 'TA', 'Gd', 'Ex'], color='blue', estimator=np.median)


# In[ ]:


sns.pointplot(x='GarageCond', y='SalePrice', data=df_train, order=['NA','Po', 'Fa', 'TA', 'Gd', 'Ex'])


# In[ ]:


sns.pointplot(x='HeatingQC', y='SalePrice', data=df_train, order=['Po', 'Fa', 'TA', 'Gd', 'Ex'], color='green', estimator=np.median)


# In[ ]:


sns.pointplot(x='ExterQual', y='SalePrice', data=df_train, order=['Po', 'Fa', 'TA', 'Gd', 'Ex'], color='orange', estimator=np.median)


# In[ ]:


sns.pointplot(x='KitchenQual', y='SalePrice', data=df_train, order=['Po', 'Fa', 'TA', 'Gd', 'Ex'], color='purple', estimator=np.median)


# In[ ]:


sns.pointplot(x='BsmtCond', y='SalePrice', data=df_train, order=['NA','Po', 'Fa', 'TA', 'Gd', 'Ex'])


# In[ ]:


sns.pointplot(x='ExterCond', y='SalePrice', data=df_train, order=['NA','Po', 'Fa', 'TA', 'Gd', 'Ex'])


# In[ ]:


sns.pointplot(x='OverallCond', y='SalePrice', data=df_train)


# In[ ]:


sns.pointplot(x='Functional', y='SalePrice', data=df_train, order=['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'])


# ## Sizing

# In[ ]:


sns.regplot(x='LotArea', y='SalePrice', data=df_train)


# In[ ]:


sns.regplot(x='1stFlrSF', y='SalePrice', data=df_train)


# In[ ]:


sns.regplot(x='2ndFlrSF', y='SalePrice', data=df_train)


# In[ ]:


sns.regplot(x='MasVnrArea', y='SalePrice', data=df_train)


# In[ ]:


sns.regplot(x='TotalBsmtSF', y='SalePrice', data=df_train)


# ## Other

# In[ ]:


sns.pointplot(x='BldgType', y='SalePrice', data=df_train)


# In[ ]:


sns.pointplot(x='HouseStyle', y='SalePrice', data=df_train, order=['1Story', '1.5Fin', '1.5Unf', '2Story', '2.5Fin', '2.5Unf', 'SFoyer', 'SLvl'], estimator=np.mean)


# In[ ]:


sns.barplot(x='MSZoning', y='SalePrice', data=df_train)


# In[ ]:


chart = sns.barplot(x='Neighborhood', y="SalePrice", data=df_train)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)


# In[ ]:


sns.pointplot(x='YrSold', y='SalePrice', data=df_train)


# # Feature Engineering

# # Preprocessing
# Before we begin modeling we have to make sure that clean our data then encode.

# ## Cleaning
# We have 35 features that have missing values and of those 35, 6 have more than half of their values missing. 23 of the 35 are categorical while the rest are numerical.

# In[ ]:


def num_clean(df_train, df_test):
    """ Clean the data before we encode"""
    #1) df_train
    df_train_num = df_train.select_dtypes(include='number') # fetch num columns
    train_missing_cols = df_train_num.columns[df_train_num.isnull().any()].tolist() # fetch num columns with missing
    df_train = _fill_num_df(df_train, train_missing_cols)
    
    # 2) df_test
    df_test_num = df_test.select_dtypes(include='number') # fetch num columns
    test_missing_cols = df_test_num.columns[df_test_num.isnull().any()].tolist() # fetch num columns with missing
    df_test = _fill_num_df(df_test, test_missing_cols)
    
    return df_train, df_test
    
def _fill_num_df(df, cols):
    """ Fill in the missing values for the dataframe """
    for col in cols:
        df[col] = df[col].fillna(df[col].mean())
    return df
    


# In[ ]:


def cat_clean(df_train, df_test):
    """ Clean the data data before we encode """
    #1) df_train
    df_train_cat = df_train.select_dtypes(include='object') # fetch cat columns
    train_missing_cols = df_train_cat.columns[df_train_cat.isnull().any()].tolist() # fetch cat columns with missing
    df_train = _fill_cat_df(df_train, train_missing_cols)
    
    # 2) df_test
    df_test_cat = df_test.select_dtypes(include='object') # fetch cat columns
    test_missing_cols = df_test_cat.columns[df_test_cat.isnull().any()].tolist() # fetch cat columns with missing
    df_test = _fill_cat_df(df_test, test_missing_cols)
    
    return df_train, df_test
    
def _fill_cat_df(df, cols):
    """ Fill in the missing values for the dataframe """
    for col in cols:
        df[col] = df[col].fillna(df[col].mode().values[0])
    return df


# ## Encoding
# Next assuming that we've successfully filled the missing values in our dataset we have to encode the **43** categorical features. In the process, let's make sure to differentiate between nominal vs ordinal categorical columns by using the corresponding encoding method. 

# In[ ]:


from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# In[ ]:


def nominal_encode(df_train, df_test, cols):
    """Encode the nominal features"""
    # create encoder
    encoder = OneHotEncoder(dtype='int', sparse=False) # sparse returns array not matrix
    # transform columns
    encoded_df_train = pd.DataFrame(encoder.fit_transform(df_train[cols])) # remove target and put back after transform
    encoded_df_test = pd.DataFrame(encoder.transform(df_test[cols]))
    # Add index back to the transformed dfs
    encoded_df_train.index = df_train.index
    encoded_df_test.index = df_test.index
    # remove the original cols b/c we're about add the encoded
    df_train = df_train.drop(cols, axis=1)
    df_test = df_test.drop(cols, axis=1)
    # create the new dfs
    df_train = pd.concat([df_train, encoded_df_train], axis=1)
    df_test = pd.concat([df_test, encoded_df_test], axis=1)
    
    return df_train, df_test
    
    
def ordinal_encode(df_train, df_test, cols):
    """Encode the ordinal features"""
    # Encoder
    encoder = OrdinalEncoder(dtype='int')
    # transform
    encoded_df_train = pd.DataFrame(encoder.fit_transform(df_train[cols]))
    encoded_df_test = pd.DataFrame(encoder.transform(df_test[cols]))
    # add index 
    encoded_df_train.index = df_train.index
    encoded_df_test.index = df_test.index
    # remove original columsn b/c we transformed them
    df_train = df_train.drop(cols, axis=1)
    df_test = df_test.drop(cols, axis=1)
    # concat
    df_train = pd.concat([df_train, encoded_df_train], axis=1)
    df_test = pd.concat([df_test, encoded_df_test], axis=1)
    
    return df_train, df_test


# In[ ]:


# 1) Clean
df_train, df_test = num_clean(df_train, df_test)
df_train, df_test = cat_clean(df_train, df_test)

# 2) Encode
df_train, df_test = nominal_encode(df_train, df_test, ["MSZoning", "Street", "Alley", "Utilities", "Exterior1st", "Exterior2nd", "MasVnrType", "MiscFeature", "SaleType", "Electrical", "GarageType", "LotShape", "LandContour", "LotConfig", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Foundation", "Heating", "PavedDrive", "CentralAir", "SaleCondition"])
df_train, df_test = ordinal_encode(df_train, df_test, ["BsmtQual", "BsmtCond", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC", "Functional", "BsmtExposure", "GarageFinish", "Fence", "LandSlope", "ExterQual", "ExterCond", "BsmtFinType1", "BsmtFinType2", "HeatingQC"])


# # Modeling

# ## Split Training Data
# Given our test dataset let's split our training data

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


TEST_SIZE = 0.25
X_all = df_train.drop(['SalePrice', 'Id'], axis=1)
y_all = df_train[['SalePrice']]

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=TEST_SIZE, random_state=3)


# ## Model Selection
# Since we are trying to predict pricing (continuous value) we should explore regressor models. Let's test several regression models and identify the one that gives us the best results. 

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression

from sklearn.metrics import max_error,mean_absolute_error,mean_squared_error,median_absolute_error,r2_score

import math


# In[ ]:


models = ['LinearRegression',
            'LinearSVR',
            'DecisionTreeRegressor',
            'GradientBoostingRegressor',
            'RandomForestRegressor']

results = {
    "models": [],
    "mean_absolute_error": [],
    "root_mean_squared_error": [],
    "median_absolute_error": [],
    "max_error": [],
    "r2_score": [],
}

for model in models:
    m = eval(model)()
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    
    results['models'].append(model)
    results['mean_absolute_error'].append(mean_absolute_error(y_test, y_pred))
    results['root_mean_squared_error'].append(math.sqrt(mean_squared_error(y_test, y_pred)))
    results['median_absolute_error'].append(median_absolute_error(y_test, y_pred))
    results['max_error'].append(max_error(y_test, y_pred))
    results['r2_score'].append(r2_score(y_test, y_pred))
                                       
results_df = pd.DataFrame(results)
results_df.sort_values(by=['root_mean_squared_error', 'r2_score'], ascending=True, inplace=True)
results_df


# We're mainly interested in the Root Mean Square Error (RMSE) since that's what the final prediction will be judged by. Looks like the winners are the **GradientBoostingRegressor** and **RandomForestRegressor** models.

# ## Tuning
# Let's try to optimize our model parameters to ensure that we get the best model.

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer


# In[ ]:


# Define scorer
def rmse_metric(y_test, y_pred):
    score = math.sqrt(mean_squared_error(y_test, y_pred))
    return score

# Scorer function would try to maximize calculated metric
rmse_scorer = make_scorer(rmse_metric, greater_is_better=False)

def tune_model(model, X, y, param_grid, cv):
    reg = GridSearchCV(estimator=eval(model)(), param_grid=param_grid, cv=cv, scoring=rmse_scorer, n_jobs=-1, verbose=False)
    reg.fit(X, y)
    return (model, reg.best_score_, reg.best_estimator_)


# In[ ]:


tuning_models = ['GradientBoostingRegressor',
                'RandomForestRegressor']

param_grid = {
    'RandomForestRegressor': {
        'n_estimators': [10, 25, 50],
        'max_depth': [10, 25, 50],
        'max_features': ['auto', 'sqrt', 'log2']
    },
    'GradientBoostingRegressor': {
        'learning_rate': [0.01, 0.001, 0.0001],
        'n_estimators': [100, 150, 200],
        'max_depth': [3, 5, 10],
        'max_features': ['auto', 'sqrt', 'log2'],
    }
}

tuned_results = {
    'model': [],
    'best_score': [],
    'best_estimator': []
}

for model in tuning_models:
    m, score, est = tune_model(model, X_train, np.ravel(y_train), param_grid[model], 5)
    tuned_results['model'].append(m)
    tuned_results['best_score'].append(score)
    tuned_results['best_estimator'].append(est)
    
tuned_results_df = pd.DataFrame(tuned_results)
tuned_results_df.sort_values(by=['best_score'], ascending=True, inplace=True)
tuned_results_df


# ## Submission

# In[ ]:


best_estimator = tuned_results_df.iloc[0,2]
y_predict = best_estimator.predict(df_test.drop(['Id'], axis=1))
y_predict_df = pd.DataFrame({'SalePrice': y_predict})

submission_df = pd.concat([df_test[['Id']], y_predict_df], axis=1)

submission_df.head()


# In[ ]:


submission_df.to_csv('Housing_Prices_Prediction_1.csv', index=False)

