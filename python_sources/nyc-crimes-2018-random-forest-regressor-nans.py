#!/usr/bin/env python
# coding: utf-8

# # NYPD Complaint - Filling NaNs with Random Forest Regressor for Beginners
# ### Data updated 7 June 2018
# 
# ### Notebook created - *2018-24-08*
# ### Version - 1
# 
# ## **This kernel is continuation of previous one**
# 
# ### **List of kernels:**
# ### 1. Previous, introductory notebook - [here](https://www.kaggle.com/mihalw28/nyc-crimes-2018-data-cleaning-part-i)
# ### 2. This one - [here](https://www.kaggle.com/mihalw28/nyc-crimes-2018-random-forest-regressor-nans)

# ## Introduction
# 
# This notebook is continuation of kernel [NYC Crimes 2018  - data cleaning, part I](https://www.kaggle.com/mihalw28/nyc-crimes-2018-data-cleaning-part-i). Like previous one, this kernel take up basic
# data science skills like data cleaning and implementing regression to fill empty values. I found many inspirations and ideas for this notebook in *Hands-On Machine Learning with Scikit_Learn and TensorFlow*  book written by [Aurelion Geron](https://twitter.com/aureliengeron). Any comments about kernel errors and better solutions are welcome. 

# ## Activities I am planning to perform in this kernel
# 
# 
# ### [FILL NAN VALUES IN PATROL_BORO COLUMN:](#1)
# 1. [First things first](#2)
# 2. [Import data & prepare for machine learning algorithms](#3)
# 3. [Train models](#4)
# 4. [Fine-tune](#5)
# 5. [Evaluation and results](#6)
# 

# <a id="1"></a> <br>
# # Fill NaN values in PATROL_BORO column

# <a id="2"></a> <br>
# ### First Things First

# In[ ]:


# Imports

# Visualisations
import matplotlib.pyplot as plt 
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')

# Warnings
import warnings
warnings.filterwarnings(action = 'ignore')

# Data exploration
import pandas as pd

# Numerical
import numpy as np

# Random
np.random.seed(11)

# Files in dataset
import os
print(os.listdir("../input"))


# <a id="3"></a> <br>
# ## Import data and prepare for machine learning algorithms

# In[ ]:


# Import data frame formed in part I of NYC crimes kernel
crimes = pd.read_csv("../input/crimes_df.csv")
crimes.info()


# In[ ]:


# Find values with NaN in PATROL_BORO column, extract them and save as a new data frame.
print("Name of the PATROL BORO: \n", crimes['PATROL_BORO'].value_counts(dropna = False), sep = '')   # check if there any NaNs
patrol_boro_nan = crimes[crimes['PATROL_BORO'].isnull()]   # df with PATROL_BORO NaNs only
patrol_boro_nan.drop('PATROL_BORO', axis = 1)   # delete PATROL_BORO column


# In[ ]:


# Create df without PATROL_BORO NaN values, to split in sets
df_p_b = crimes.dropna(subset = ['PATROL_BORO'], axis = 0).reset_index()   # reset_index() is crucial here
# Sanity check
df_p_b['PATROL_BORO'].value_counts(dropna = False)


# In[ ]:


# Split data in train and test set. Use StratifiedShuffleSplit to make low (lower than splitting with purely random values)
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 11)
for train_index, test_index in split.split(df_p_b, df_p_b['PATROL_BORO']):
    strat_train_set = df_p_b.loc[train_index]
    strat_test_set = df_p_b.loc[test_index]


# In[ ]:


# Check values in test set
#print(strat_test_set['PATROL_BORO'].value_counts(normalize = True))


# In[ ]:


# Create df with crimes/incidents labels of train set, and drop PATROL_BORO column (maybe not necessary to drop column, because I pick categorical columns manually)
crimes_labels = strat_train_set['PATROL_BORO'].copy().to_frame()
crimes = strat_train_set.drop('PATROL_BORO', axis = 1)


# In[ ]:


# Quick plot for data check
crimes.plot(kind = 'scatter', x = 'Longitude', y = 'Latitude', marker = 'o', alpha = 0.08, figsize = (16,12));


# In[ ]:


# Select categories to feed the model, all numerical without index and one categoroical.
# To be honest I didn`t wonder much time what to select from categorical series, but BORO_NM should be a perfect match
crimes_num = crimes.select_dtypes(include = [np.number]).drop('index', axis = 1)
crimes_cat = crimes['BORO_NM']


# Application of SimpleImputer to fill NaNs in numerical values is a bit useless here, because there aren't any NaNs. Nevertheless, it is very usefull if the data set was not cleaned up before.

# In[ ]:


# Deal with numerical NaNs 
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(crimes_num)
imputer.transform(crimes_num)


# In[ ]:


# Encode crimes labels, use OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse = False)
crimes_labels_1hot = onehot_encoder.fit_transform(crimes_labels)
print(crimes_labels_1hot.shape)
crimes_labels_1hot


# The next step is to write a custom transformer to automatically feed a pipeline with selected numerical or categorical attributes. Source [here](https://github.com/ageron/handson-ml/blob/master/02_end_to_end_machine_learning_project.ipynb).

# In[ ]:


# Write a selector 
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


# In[ ]:


# Make pipelines for numerical and categorical attributes
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_attribs = list(crimes_num)
cat_attribs = ['BORO_NM']

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('cat_encoder', OneHotEncoder(sparse=False)), 
])


# In[ ]:


# Create one pipeline for the whole process
from sklearn.pipeline import FeatureUnion
full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline),
])


# In[ ]:


# Encode values using full_pipeline
crimes_prepared = full_pipeline.fit_transform(crimes)
print(crimes_prepared.shape)
crimes_prepared


# <a id="4"></a> <br>
# ## Train models

# Time to select and train machine learning model.

# ### **Linear Regression**

# In[ ]:


# Linear regression model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(crimes_prepared, crimes_labels_1hot) #return to crimes_labels_encoded


# In[ ]:


from sklearn.metrics import mean_squared_error
crimes_predictions = lin_reg.predict(crimes_prepared)
lin_mse = mean_squared_error(crimes_labels_1hot, crimes_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[ ]:


from sklearn.metrics import mean_absolute_error

lin_mae = mean_absolute_error(crimes_labels_1hot, crimes_predictions)
lin_mae


# ### **Decision Tree**

# In[ ]:


# Decsision tree regressor model
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(random_state=11)
tree_reg.fit(crimes_prepared, crimes_labels_1hot)


# In[ ]:


# Don't use code from this cell to predict labels. Data overfitted - too good to be true.
# Uncomment below to check rsme.

# crimes_predictions = tree_reg.predict(crimes_prepared)
# tree_mse = mean_squared_error(crimes_labels_1hot, crimes_predictions)
# tree_rmse = np.sqrt(tree_mse)
# print("Decision Tree Regressor: rmse:", tree_rmse) 


# In[ ]:


# Better option, cross validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, crimes_prepared, crimes_labels_1hot, scoring = 'neg_mean_squared_error', cv = 10)
tree_rmse_scores = np.sqrt(-scores)


# In[ ]:


# Display all scores
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# In[ ]:


# Execute display_scores(scores) function
display_scores(tree_rmse_scores)


# In[ ]:


# Compute the same scores for Linear Regression
lin_scores = cross_val_score(lin_reg, crimes_prepared, crimes_labels_1hot, scoring = 'neg_mean_squared_error', cv = 10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# ### **Random forest**

# In[ ]:


# Random forset Regressor model
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(random_state=11)
forest_reg.fit(crimes_prepared, crimes_labels_1hot)

crimes_predictions = forest_reg.predict(crimes_prepared)
forest_mse = mean_squared_error(crimes_labels_1hot, crimes_predictions)
forest_rmse = np.sqrt(forest_mse)
print("Random Forest Regressor -> rmse:", forest_rmse)


# In[ ]:


# Compute cross_val_score for Random Forest Regressor
from sklearn.model_selection import cross_val_score
forest_scores = cross_val_score(forest_reg, crimes_prepared, crimes_labels_1hot, scoring='neg_mean_squared_error', cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


# <a id="5"></a> <br>
# ## **Fine-tune model**

# ### **Grid search**

# In[ ]:


# Grid search using RFR
from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [3, 4, 5, 6]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
forest_reg = RandomForestRegressor(random_state = 11)
grid_search = GridSearchCV(forest_reg, param_grid, cv = 5, scoring = 'neg_mean_squared_error', 
                          return_train_score = True)
grid_search.fit(crimes_prepared, crimes_labels_1hot)
print("Grid search best parameters: ", grid_search.best_params_)
print("Grid search best estimator: ", grid_search.best_estimator_)


# In[ ]:


# Evaluation scores
print("Evaluation scores")
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)


# ### **Randomized search**

# In[ ]:


# Randomized search on RFR
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
    'n_estimators': randint(low = 1, high = 200),
    'max_features': randint(low = 1, high = 8),
}

forest_reg = RandomForestRegressor(random_state = 11)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions = param_distribs,
                               n_iter = 10, cv = 5, scoring = 'neg_mean_squared_error', random_state = 11)
rnd_search.fit(crimes_prepared, crimes_labels_1hot)


# In[ ]:


feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# In[ ]:


# Check most important attributes
cat_encoder = cat_pipeline.named_steps['cat_encoder']
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse = True)


# <a id="6"></a> <br>
# ## **Evaluation and results**

# In[ ]:


# Evaluate model on test set
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop('PATROL_BORO', axis = 1)
y_test = strat_test_set['PATROL_BORO'].copy().to_frame()

# Second step - OneHotEncoder, enoding integers to sparse matrix as an output, if (sparse = False) array as an output
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder(sparse = False)
y_test_encoded_oh = cat_encoder.fit_transform(y_test)
y_test_encoded_oh

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_predictions
final_mse = mean_squared_error(y_test_encoded_oh, final_predictions)
final_rmse = np.sqrt(final_mse)
print("Final score:", final_rmse)


# In[ ]:


# Find PATROL_BORO NaN values. Evaluate final model on patrol_boro_nan data frame
X_to_find = full_pipeline.transform(patrol_boro_nan)
NaNs_found = final_model.predict(X_to_find)
NaNs_found[:5]


# In[ ]:


# Decode values
# decode one hot oncoder
one_hot_decode = cat_encoder.inverse_transform(NaNs_found)
one_hot_decode[:5]


# In[ ]:


# Make data frame of founded NaNs and fix index
found = pd.DataFrame(one_hot_decode, columns = ['PATROL_BORO'], index = patrol_boro_nan.index)
found[:5]


# In[ ]:


# Read original data frame
crimes_original = pd.read_csv("../input/crimes_df.csv")
crimes_original['PATROL_BORO'].value_counts(dropna = False)


# In[ ]:


# Fill crimes_original PATROL_BORO NaNs with found values
for index in crimes_original['PATROL_BORO'], found['PATROL_BORO']:
    crimes_original['PATROL_BORO'].loc[crimes_original['PATROL_BORO'].isnull()] = found['PATROL_BORO']


# In[ ]:


# Check
crimes_original.info()


# In[ ]:


# Write df to csv
crimes_original.to_csv('crimes_NYC.csv', index = False)


# In[ ]:




