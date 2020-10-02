#!/usr/bin/env python
# coding: utf-8

# Notebook created: 2018-01-10
# Last update: 2018-01-10
# Version 1
# 
# List of previous kernels relative to data set used in this kernel:
# 1. First, introductory kernel - [NYC Crimes 2018 - data cleaning, part I](https://www.kaggle.com/mihalw28/nyc-crimes-2018-data-cleaning-part-i)
# 2. Second - [NYC Crimes 2018 - Random Forest Regressor & NaNs](https://www.kaggle.com/mihalw28/nyc-crimes-2018-random-forest-regressor-nans)
# 3. This one - [Fill NaNs using regression, part II](https://www.kaggle.com/mihalw28/fill-nans-with-ml)
# 4. Last, mostly visualisations - [NYC crimes 2018 - visualistions](https://www.kaggle.com/mihalw28/nyc-crimes-2018-visualistions)

# Short table of contents:
# 1. [Imports](#1)
# 2. ['SUSP_RACE':](#2)
#     * [Train models](#3)
#     * [Fine-tune](#4)
#     * [Evaluation](#5)
# 3. ['SUSP_AGE_GROUP'](#6)
#     * [Train models](#7)
#     * [Fine-tune](#8)
#     * [Evaluation](#9)
# 4. ['SUSP_SEX'](#10)
#     * [Train models](#11)
#     * [Fine-tune](#12)
#     * [Evaluation](#13)
# 5. [Visual comparisons](#14)
# 6. [Final toughts](#15)

# This notebook is continuation of previous two. To understand the context I invite to read first and second relative to the same data set. In this kernel I try to improve method of filling NaN values which I have applied in [NYC Crimes 2018 - Random Forest Regressor & NaNs](https://www.kaggle.com/mihalw28/nyc-crimes-2018-random-forest-regressor-nans) notebook through apply functions to reduce wasting time. What is more, at the end of this notebook I present some comparisons between filling NaNs using random values and regression. 

# <a id="1"></a> <br>
# # Imports

# In[ ]:


# Visualisations
import matplotlib.pyplot as plt 
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly
import plotly.offline as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from plotly.tools import make_subplots
init_notebook_mode()

# Warnings
import warnings
warnings.filterwarnings(action = 'ignore')

# Data exploration
import pandas as pd

# Numerical
import numpy as np

# Random
np.random.seed(11)

# Splitt and encode
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder

# Make pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion

# Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

# Grid Search
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Files in dataset
import os
print(os.listdir("../input"))


# In[ ]:


# Load and inspect data
crimes_df = pd.read_csv("../input/crimes_NYC.csv")
crimes_df.info()


# As you can see all columns in data frame have 109543 non-null values except 3. *SUSP_AGE_GROUP*, *SUSP_RACE*, *SUSP_SEX* with filled NaNs respectively *suspector_age_rand*, *suspector_race_rand* and *suspector_sex_rand* are placed at the end of the df. It is important to mention that already filled suspector columns are filled with random values. More info in previous [kernel](https://www.kaggle.com/mihalw28/nyc-crimes-2018-random-forest-regressor-nans).
# 
# To be precise, not all columns in data frame have appropriate values. If you want verificate it yourself, chek values in victim columns. However I didn't have reason to change those values yet. (There is a possibility I'll take care of it later.)

# <a id="2"></a> <br>
# # SUSP_RACE

# In[ ]:


# Make df with NaNs in target column (SUSP_RACE here) and return df without that column
def susp_n(column_name, condition):
    print("Values in {}: \n".format(column_name), crimes_df[column_name].value_counts(dropna = False), sep = '')   # check if there any NaNs
    crimes_df.loc[(condition), column_name] = np.nan # change 'WRONG' values to NaNs
    susp_nan = crimes_df[crimes_df[column_name].isnull()] # df with NaNs only
    susp_nan.drop(column_name, axis = 1)   # delete SUSP_RACE column
    print("\nValues in {} after changing: \n".format(column_name), susp_nan[column_name].value_counts(dropna = False), sep = '')   # check if there any NaNs
    return susp_nan


# In[ ]:


# Select column and condition to change values
column_name = 'SUSP_RACE'
condition = crimes_df[column_name] == 'UNKNOWN'
sn = susp_n(column_name, condition)


# In[ ]:


# Split data into train and test sets using StratifiedShuffleSplit
def for_split(column_name):
    df_non_nan = crimes_df.dropna(subset = [column_name], axis = 0).reset_index() #reset_index() is crucial here
    split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 11)
    for train_index, test_index in split.split(df_non_nan, df_non_nan[column_name]):
        strat_train_set = df_non_nan.loc[train_index]
        strat_test_set = df_non_nan.loc[test_index]
    return strat_train_set, strat_test_set


# In[ ]:


# Execute for_split()
strat_train_set, strat_test_set = for_split(column_name)


# In[ ]:


# Make train df and extract target labels
def train_and_drop(column_name, strat_train_set):
    susp_labels = strat_train_set[column_name].copy().to_frame()
    crimes_df = strat_train_set.drop(column_name, axis = 1)
    return susp_labels, crimes_df


# In[ ]:


# Execute train_and_drop
susp_labels, crimes_df = train_and_drop(column_name, strat_train_set)


# In[ ]:


# Select categorical and numerical values to feed pipeline
def select(selected_list):
    crimes_df_num = crimes_df.select_dtypes(include = [np.number]).drop('index', axis = 1)
    crimes_df_cat = crimes_df[selected_list]
    return crimes_df_num, crimes_df_cat


# In[ ]:


selected_list = ['BORO_NM'] # list of categorical values
crimes_df_num, crimes_df_cat = select(selected_list)


# In[ ]:


# Encode categorical labels
def one_hot_encoder(labels):
    one_hot_encoder = OneHotEncoder(sparse = False)
    crimes_labels_1hot = one_hot_encoder.fit_transform(labels) # for scikit-learn versions 19.x this line won`t work, because susp_race_labels are not int type
    return crimes_labels_1hot


# In[ ]:


# Execute encoding
crimes_labels_1hot = one_hot_encoder(susp_labels)
print(crimes_labels_1hot)


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


# One pipeline for cat and num values
def pipeline(df):
    num_attribs = list(crimes_df_num)
    cat_attribs = list(crimes_df_cat)
    
    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        #('imputer', SimpleImputer(strategy="median")), SimpleImputer won't work here, all num_attribs are already completed and don't have any NaN
        ('std_scaler', StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat_encoder', OneHotEncoder(sparse = False)), 
    ])
    
    # Create one pipeline for the whole process
    full_pipeline = FeatureUnion(transformer_list = [
        ('num_pipeline', num_pipeline),
        ('cat_pipeline', cat_pipeline),
    ])
    
    # Encode values using full_pipeline
    crimes_prepared = full_pipeline.fit_transform(crimes_df)
    
    return crimes_prepared, num_pipeline, cat_pipeline, num_attribs, full_pipeline


# In[ ]:


# Execute pipeline()
crimes_prepared, num_pipeline, cat_pipeline, num_attribs, full_pipeline = pipeline(crimes_df) # Is it ok, to load all these values from function?


# <a id="3"></a> <br>
# ## Train models

# ### Linear Regression

# In[ ]:


# Compute linear regression
def linear_reg(crimes_prepared, crimes_labels_1hot):
    lin_reg = LinearRegression()
    lin_reg.fit(crimes_prepared, crimes_labels_1hot)
    
    #compute with cross_cal_score
    lin_scores = cross_val_score(lin_reg, crimes_prepared, crimes_labels_1hot, scoring = 'neg_mean_squared_error', cv = 10)
    lin_rmse_scores = np.sqrt(-lin_scores)
    return lin_rmse_scores


# In[ ]:


# Execute lin_reg()
lin_rmse_scores = linear_reg(crimes_prepared, crimes_labels_1hot)


# ### Decision Tree
# 

# In[ ]:


# Decission tree regression
def decision_t(crimes_prepared, crimes_labels_1hot):
    tree_reg = DecisionTreeRegressor(random_state=11)
    tree_reg.fit(crimes_prepared, crimes_labels_1hot)
    
    # compute with cross_val_score
    scores = cross_val_score(tree_reg, crimes_prepared, crimes_labels_1hot, scoring = 'neg_mean_squared_error', cv = 10)
    tree_rmse_scores = np.sqrt(-scores)
    
    return tree_rmse_scores


# In[ ]:


tree_rmse_scores = decision_t(crimes_prepared, crimes_labels_1hot)


# In[ ]:


# Display all scores
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# In[ ]:


# Execute display_scores(scores) function
print("Linear Regression")
display_scores(lin_rmse_scores)
print("\nDecision Tree")
display_scores(tree_rmse_scores)


# ### Random Forest

# In[ ]:


# Calculate rf score
def random_f(crimes_prepared, crimes_labels_1hot):
    forest_reg = RandomForestRegressor(random_state=11)
    forest_reg.fit(crimes_prepared, crimes_labels_1hot)
    
    # compute with cross_val_score
    scores = cross_val_score(forest_reg, crimes_prepared, crimes_labels_1hot, scoring = 'neg_mean_squared_error', cv = 10)
    forest_rmse_scores = np.sqrt(-scores)
    return forest_rmse_scores


# In[ ]:


forest_rmse_scores = random_f(crimes_prepared, crimes_labels_1hot)


# In[ ]:


# Execute display_scores(scores) function
print("Random forest")
display_scores(forest_rmse_scores)


# <a id="4"></a> <br>
# ## Fine-tune model

# ### Grid Search

# In[ ]:


# Find best params using Grid Search
def grid():
    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [3, 5, 7, 9]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]
    forest_reg = RandomForestRegressor(random_state = 11)
    grid_search = GridSearchCV(forest_reg, param_grid, cv = 5, scoring = 'neg_mean_squared_error', 
                          return_train_score = True)
    grid_search.fit(crimes_prepared, crimes_labels_1hot)
    return grid_search


# In[ ]:


# Execute grid() function
grid_search = grid()


# In[ ]:


print("Grid search best parameters: ", grid_search.best_params_)
print("Grid search best estimator: ", grid_search.best_estimator_)


# In[ ]:


# Evaluation scores
print("Evaluation scores")
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)


# ### Randomized search

# In[ ]:


# Find best params using Randomized Search
def random_search():
    param_distribs = {
        'n_estimators': randint(low = 1, high = 200),
        'max_features': randint(low = 3, high = 9),
    }
    forest_reg = RandomForestRegressor(random_state = 11)
    rnd_search = RandomizedSearchCV(forest_reg, param_distributions = param_distribs,
                                    n_iter = 10, cv = 5, scoring = 'neg_mean_squared_error', random_state = 11)
    rnd_search.fit(crimes_prepared, crimes_labels_1hot)

    return rnd_search


# In[ ]:


# Execute random_search()
rnd_search = random_search()


# In[ ]:


print("Best parameters: {}".format(rnd_search.best_params_))
print("Best random search score: {}".format(np.sqrt(-rnd_search.best_score_)))


# In[ ]:


# find feature importance
def important_feat():
    feature_importances = grid_search.best_estimator_.feature_importances_
    # Chcek most importanct attributes
    cat_encoder = cat_pipeline.named_steps['cat_encoder']
    cat_one_hot_attribs = list(cat_encoder.categories_[0])
    attributes = num_attribs + cat_one_hot_attribs
    return sorted(zip(feature_importances, attributes), reverse = True)


# In[ ]:


# Execute important_feat()
feature_importances = important_feat()
print(feature_importances)


# <a id="5"></a> <br>
# ## Evaluation

# In[ ]:


# Evaluate final score on test set
def final_score_eval(column_name):
    final_model = grid_search.best_estimator_
    
    X_test = strat_test_set.drop(column_name, axis = 1)
    y_test = strat_test_set[column_name].copy().to_frame()
    
    # Second step - OneHotEncoder, encoding integers to sparse matrix as an output, if (sparse = False) array as an output
    cat_encoder = OneHotEncoder(sparse = False)
    y_test_encoded_ohe = cat_encoder.fit_transform(y_test)
    
    X_test_prepared = full_pipeline.transform(X_test)
    final_predictions = final_model.predict(X_test_prepared)
    
    final_mse = mean_squared_error(y_test_encoded_ohe, final_predictions)
    final_rmse = np.sqrt(final_mse) 
    print("Final score: ", final_rmse)
    return final_rmse, final_model, cat_encoder


# In[ ]:


# Execute
final_rmse, final_model, cat_encoder= final_score_eval(column_name)


# Frankly, it isn't the best performance I've ever seen, and it's slightly lower than best training score. Is it kind of underfitting?

# In[ ]:


def find_nans(sn):
   X_to_find = full_pipeline.transform(sn)
   NaNs_found = final_model.predict(X_to_find)
   return NaNs_found


# In[ ]:


# Execute find_nans()
NaNs_found = find_nans(sn)
NaNs_found[:5]


# In[ ]:


# Decode values, encoded with OneHotEncoder
one_hot_decode = cat_encoder.inverse_transform(NaNs_found)
one_hot_decode[:5]


# In[ ]:


# Make a df with decoded values
def nans_df(column_name):
    found = pd.DataFrame(one_hot_decode, columns = [column_name], index = sn.index)
    return found


# In[ ]:


# Execute nans_df 
found_race = nans_df(column_name)
found_race[:5]


# In[ ]:


# Fill NaNs in raw data frame
def fill_nans(col_df, data_frame):
    for index in col_df[column_name].index, data_frame[column_name].index:
        data_frame[column_name].loc[data_frame[column_name].isnull()] = col_df[column_name]
    return data_frame


# In[ ]:


# Make a room for new values in raw dataframe
def make_room(df, raw_condition):
    df.loc[raw_condition, column_name] = np.nan
    return df


# In[ ]:


# Read original data frame
crimes_NYC_raw = pd.read_csv("../input/crimes_NYC.csv")
crimes_NYC_raw[column_name].value_counts(dropna = False)
raw_condition = crimes_NYC_raw[column_name] == 'UNKNOWN'
crimes_NYC = make_room(crimes_NYC_raw, raw_condition)


# In[ ]:


fill_nans(found_race, crimes_NYC)


# In[ ]:


# Uncomment for sanity check
crimes_NYC.info()
crimes_NYC['SUSP_RACE'].value_counts(dropna = False)


# As you could see my models are pretty simple. I selected basic regression models and followed them to get scores. I  won't intend find new models' parameters and try to obtain better results in this project. Or maybe if I'll find while. :)

# <a id="6"></a> <br>
# # SUSP_AGE_GROUP

# In[ ]:


# Copy crimes_NYC df
crimes_df = crimes_NYC.copy()


# In[ ]:


# Start values for 'SUSP_AGE_GROUP' column
column_name = 'SUSP_AGE_GROUP'
condition = ((crimes_df[column_name] != '25-44') & 
               (crimes_df[column_name] != '18-24') &
               (crimes_df[column_name] != '45-64') &
               (crimes_df[column_name] != '65+') &
               (crimes_df[column_name] != '<18'))


# In[ ]:


sn = susp_n(column_name, condition)


# In[ ]:


sn['SUSP_AGE_GROUP'].value_counts(dropna = False)


# In[ ]:


strat_train_set, strat_test_set = for_split(column_name)


# In[ ]:


susp_labels, crimes_df = train_and_drop(column_name, strat_train_set)


# In[ ]:


selected_list = ['BORO_NM', 'SUSP_RACE']
crimes_df_num, crimes_df_cat = select(selected_list)


# In[ ]:


crimes_labels_1hot = one_hot_encoder(susp_labels)
print(crimes_labels_1hot)


# In[ ]:


crimes_prepared, num_pipeline, cat_pipeline, num_attribs, full_pipeline = pipeline(crimes_df)


# <a id="7"></a> <br>
# ## Train models

# ### Linear Regression

# In[ ]:


lin_rmse_scores = linear_reg(crimes_prepared, crimes_labels_1hot)


# ### Decision Tree

# In[ ]:


tree_rmse_scores = decision_t(crimes_prepared, crimes_labels_1hot)


# In[ ]:


# Execute display_scores(scores) function
print("Linear Regression")
display_scores(lin_rmse_scores)
print("\nDecision Tree")
display_scores(tree_rmse_scores)


# ### Random Forest

# In[ ]:


forest_rmse_scores = random_f(crimes_prepared, crimes_labels_1hot)


# In[ ]:


# Execute display_scores(scores) function
print("Random forest")
display_scores(forest_rmse_scores)


# <a id="8"></a> <br>
# ## Fine-tune model

# ### Grid search

# In[ ]:


grid_search = grid()


# In[ ]:


print("Grid search best parameters: ", grid_search.best_params_)
print("Grid search best estimator: ", grid_search.best_estimator_)


# In[ ]:


# Evaluation scores
print("Evaluation scores")
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)


# ### Randomized search

# In[ ]:


rnd_search = random_search()


# In[ ]:


print("Best parameters: {}".format(rnd_search.best_params_))
print("Best random search score: {}".format(np.sqrt(-rnd_search.best_score_)))


# In[ ]:


feature_importances = important_feat()
print(feature_importances)


# <a id="9"></a> <br>
# ## Evaluation

# In[ ]:


final_rmse, final_model, cat_encoder= final_score_eval(column_name)


# In[ ]:


NaNs_found = find_nans(sn)
NaNs_found[:5]


# In[ ]:


# Decode values, encoded with OneHoTEncoder
one_hot_decode = cat_encoder.inverse_transform(NaNs_found)
one_hot_decode[:5]


# In[ ]:


found_age = nans_df(column_name)
found_age[:5]


# In[ ]:


# Read original data frame
#crimes_NYC = pd.read_csv("../input/nyc-crimes-2018-random-forest-regressor-nans/crimes_NYC.csv")
crimes_NYC[column_name].value_counts(dropna = False)

raw_condition = ((crimes_NYC[column_name] != '25-44') & 
               (crimes_NYC[column_name] != '18-24') &
               (crimes_NYC[column_name] != '45-64') &
               (crimes_NYC[column_name] != '65+') &
               (crimes_NYC[column_name] != '<18'))

crimes_NYC = make_room(crimes_NYC, raw_condition)

fill_nans(found_age, crimes_NYC)


# In[ ]:


crimes_NYC.info()


# <a id="10"></a> <br>
# # SUSP_SEX

# In[ ]:


# copy crimes_NYC df
crimes_df = crimes_NYC.copy()


# In[ ]:


# Start values for 'SUSP_SEX'
column_name = 'SUSP_SEX'
condition = ((crimes_df[column_name] != 'F') & 
              (crimes_df[column_name] != 'M'))


# In[ ]:


sn = susp_n(column_name, condition)


# In[ ]:


sn['SUSP_SEX'].value_counts(dropna=False)


# In[ ]:


strat_train_set, strat_test_set = for_split(column_name)


# In[ ]:


susp_labels, crimes_df = train_and_drop(column_name, strat_train_set)


# In[ ]:


selected_list = ['BORO_NM', 'SUSP_RACE', 'SUSP_AGE_GROUP']
crimes_df_num, crimes_df_cat = select(selected_list)


# In[ ]:


crimes_labels_1hot = one_hot_encoder(susp_labels)
print(crimes_labels_1hot)


# In[ ]:


crimes_prepared, num_pipeline, cat_pipeline, num_attribs, full_pipeline = pipeline(crimes_df)


# <a id="11"></a> <br>
# ## Train Models

# ### Liear Regression

# In[ ]:


lin_rmse_scores = linear_reg(crimes_prepared, crimes_labels_1hot)


# ### Decission Tree

# In[ ]:


tree_rmse_scores = decision_t(crimes_prepared, crimes_labels_1hot)


# In[ ]:


# Execute display_scores(scores) function
print("Linear Regression")
display_scores(lin_rmse_scores)
print("\nDecision Tree")
display_scores(tree_rmse_scores)


# ### Random Forest

# In[ ]:


forest_rmse_scores = random_f(crimes_prepared, crimes_labels_1hot)


# In[ ]:


# Execute display_scores(scores) function
print("Random forest")
display_scores(forest_rmse_scores)


# <a id="12"></a> <br>
# ## Fine-tune model

# ### Grid Search

# In[ ]:


grid_search = grid()


# In[ ]:


print("Grid search best parameters: ", grid_search.best_params_)
print("Grid search best estimator: ", grid_search.best_estimator_)


# In[ ]:


# Evaluation scores
print("Evaluation scores")
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)


# ### Randomized Search
# 

# In[ ]:


rnd_search = random_search()


# In[ ]:


print("Best parameters: {}".format(rnd_search.best_params_))
print("Best random search score: {}".format(np.sqrt(-rnd_search.best_score_)))


# In[ ]:


feature_importances = important_feat()
print(feature_importances)


# <a id="13"></a> <br>
# ## Evaluation

# In[ ]:


final_rmse, final_model, cat_encoder= final_score_eval(column_name)


# In[ ]:


NaNs_found = find_nans(sn)
NaNs_found[:5]


# In[ ]:


# Decode values, encoded with OneHoTEncoder
one_hot_decode = cat_encoder.inverse_transform(NaNs_found)
one_hot_decode[:5]


# In[ ]:


found_sex = nans_df(column_name)
found_sex[:6]


# In[ ]:


crimes_NYC[column_name].value_counts(dropna = False)

raw_condition = ((crimes_NYC[column_name] != 'F') & 
              (crimes_NYC[column_name] != 'M'))
crimes_NYC = make_room(crimes_NYC, raw_condition)
fill_nans(found_sex, crimes_NYC)


# In[ ]:


crimes_NYC.info()


# In[ ]:


# Write df to csv
crimes_NYC.to_csv('crimes_complete.csv', index = False)


# <a id="14"></a> <br>
# # Visual comparisons

# ### Prepare data

# In[ ]:


# Find rows in main data frame with values filled in selected column
def find_rows(df_with_index, selected_df):
    crimes_filtered = crimes_NYC.iloc[crimes_NYC.index.isin(selected_df.index)]
    return crimes_filtered


# In[ ]:


# Execute
# find race
df_filled_race = find_rows(crimes_NYC, found_race)
# find age
df_filled_age = find_rows(crimes_NYC, found_age)
# find sex
df_filled_sex = find_rows(crimes_NYC, found_sex)


# In[ ]:


# Make comparison of values in random values columns and regression based values
df_filled_race = df_filled_race[['SUSP_RACE', 'suspector_race_rand']]
df_filled_race.reset_index()
df_filled_age = df_filled_age[['SUSP_AGE_GROUP', 'suspector_age_rand']]
df_filled_age.reset_index()
df_filled_sex = df_filled_sex[['SUSP_SEX', 'suspector_sex_rand']]
df_filled_sex.reset_index()


# In[ ]:


race_reg = df_filled_race['SUSP_RACE'].value_counts(normalize = True) * 100
race_rnd = df_filled_race['suspector_race_rand'].value_counts(normalize = True) * 100
print(race_reg.index)
age_reg = df_filled_age['SUSP_AGE_GROUP'].value_counts(normalize = True) * 100
age_rnd = df_filled_age['suspector_age_rand'].value_counts(normalize = True) * 100
print(age_reg.index)
sex_reg = df_filled_sex['SUSP_SEX'].value_counts(normalize = True) * 100
sex_rnd = df_filled_sex['suspector_sex_rand'].value_counts(normalize = True) * 100
print(sex_reg.index)


# In[ ]:


# Make list from all series to obtain x values in right format (list of lists)
lista = [race_reg, race_rnd, age_reg, age_rnd, sex_reg, sex_rnd]

# Function takes list of series and returns proper x values for chart
def make_list(lista):
    lista2 = []
    for item in lista:
        item = list(item)
        lista2.append(item)
    return lista2


# In[ ]:


# Execute function make_lists
x = make_list(lista)
x_race = x[0:2]
x_sex = x[4:6]
x_age = x[2:4]
all_x = [x_race, x_age, x_sex] # just to make easier looping through 
indexes = [race_reg.index, age_reg.index, sex_reg.index]
#print(x_race)
#print(x_data[0])
print(all_x[0])
print(all_x)
print(x[0])


# In[ ]:


# Function to select visible traces at start
def vis(value):
    if value == 0:
        visible = True
    else:
        visible = False
    return visible


# In[ ]:


# Function to group legends - kind of hardcoding 
def legend_gr(count):
    # find group
    if float(count * 0.5).is_integer():
        legend_group = 'group' + str((int(count * 0.5) + 1))
    else:
        legend_group = 'group' + str(int((count + 1) * 0.5))
    return legend_group


# In[ ]:


# Function to hide duplicated legends
def hide(count):
    if count % 2 == 0:
        showlegend = True
    else:
        showlegend = False
    return showlegend


# In[ ]:


# Find percent of found values

colors = ['#154B68', '#166D88',
          '#38686A', '#869485',
          '#BBB49F',  '#B1B6B9']


x_data = x
y_data = ['Values from<br>regression', 'Random values']


# traces
traces = []
count = 0
for j in range(0, len(all_x)):
    for i in range(0, len(all_x[j][0])):
        for xd, yd in zip(all_x[j], y_data):
            traces.append(go.Bar(
                x = [xd[i]],
                y = [yd],
                text = ["{0:.2f}".format(xd[i]) + '%'],
                hoverinfo = 'text',
                orientation = 'h',
                showlegend = hide(count),
                name = indexes[j][i],
                visible = vis(j),
                legendgroup = str(legend_gr(count)),
                marker = dict(
                    color = colors[i],
                    line = dict(
                        color = '#f8f8f8',
                        width = 1
                    )
                )
            ))
            count += 1


updatemenus = list([
    dict(active=-1,
         buttons=list([   
            dict(label = 'race',
                method = 'update',
                args = [{'visible': [True, True, True, True, True, True, True, True, True, True, True, True, 
                                     False, False, False, False, False, False, False, False, False, False,
                                     False, False, False, False]},
                        {'title': 'Percentage differences between ways of filling in NaN values<br>in suspector race column.'}]),
            dict(label = 'age',
                method = 'update',
                args = [{'visible': [False, False, False, False, False, False, False, False, False, False, False, False,
                                     True, True, True, True, True, True, True, True, True, True,
                                     False, False, False, False]},
                        {'title': 'Percentage differences between ways of filling in NaN values<br>in suspector age group column.'}]),
            dict(label = 'sex',
                method = 'update',
                args = [{'visible': [False, False, False, False, False, False, False, False, False, False, False, False,
                                     False, False, False, False, False, False, False, False, False, False,
                                     True, True, True, True]},
                        {'title': 'Percentage differences between ways of filling in NaN values<br>in suspector sex column.'}]),
        ]),
        pad = {'r': 0, 't': 2},
        x = -0.01,
        y = 1.17,
        yanchor = 'top',
        direction = 'right',
        bgcolor = '#daecf8',
        bordercolor = '#daecf8',
        font = dict(size=11, color='#2f2f30')
    )
])

layout = go.Layout(
    title = 'Percentage differences between ways of filling in NaN values',
    xaxis=dict(
        showgrid = True,
        showline = True,
        autorange = True,
        #rangemode = 'normal',
        #fixedrange = True,
        linecolor = '#D7DEE2',
        showticklabels = True,
        ticklen = 8,
        tickwidth = 1,
        tickcolor='#D7DEE2',
        ticks='outside',
        zeroline = False, 
        domain = [0.15, 1],
        titlefont = dict(
            family = 'Arial, sans-serif',
            size = 18,
            color = '#2f2f30'
        ),
    ),
    yaxis = dict(
        showgrid = False,
        showline = False,
        showticklabels = False,
        zeroline = False,
    ),
    barmode = 'stack',
    bargap = 0.3,
    paper_bgcolor = '#EFF7FC',
    plot_bgcolor = '#EFF7FC',
    margin = dict(
        l = 150,
        r = 185,
        t = 140,
        b = 100
    ),
    height = 500,
    width = 1000,
    updatemenus=updatemenus,
    legend = dict(
        font=dict(
            family = 'Arial, sans-serif',
            size = 12,
            color = '#2f2f30'
        )
    )
)

annotations = list([
    dict(text = 'Select<br>feature', 
         x = -0.2,
         y = 1.17,
         yref = 'paper',
         xref = 'paper',
         align = 'left',
         showarrow = False,
         font = dict(
             size = 14,
             color = '#2f2f30'))
])

for yd, xd in zip(y_data, x_data):
    # labeling the y-axis
    annotations.append(dict(xref = 'paper', yref = 'y',
                            x = 0.11, y = yd,
                            xanchor = 'right',
                            text = str(yd),
                            font = dict(family = 'Arial, sans-serif', size = 16,
                                      color = '#2f2f30'),
                            showarrow = False, align = 'right'))

layout['annotations'] = annotations

fig = go.Figure(data=traces, layout=layout)
py.iplot(fig)


# <a id="15"></a> <br>
# # Summary and final toughts

# 1. Rising rmse in filled columns, could be caused by many factors like decreasing amount of training data or caused by basing on filled columns, not on raw data.
# 2. There are visible differences between columns using random values and regression values. Training sets equal to 50% of all data are too small of course.
# 
# Thank you for your time.
# If you know better solutions for this kind of problem don't hesitate to post it. Any comments are welcome.

# In[ ]:




