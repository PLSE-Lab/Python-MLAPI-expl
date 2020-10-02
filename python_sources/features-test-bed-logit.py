#!/usr/bin/env python
# coding: utf-8

# # Pipeline-based framework for features evaluation
# 
# This approach allows easily adding new features and trying different combinations of features and model parameters. It's possible to run grid search over combinations of features.
# 
# The general idea:
#   * First pipeline processes the data and creates dataset with extracted features
#   * Second pipeline allows to keep only selected features and fit the model
#   * Grid search is done with the 2nd pipeline
#   * Get predictions from 2nd pipeline with the best parameters
# 
# Downsides & possible improvements:
#   * Grid search only applies to features list and model parameters but not to pre-processing pipeline (as an ugly workaround, it's possible to create multiple features with different params, e.g. multiple TfidfVectorizer with different ngram param, and test them out)
#   * Grid search can't be scaled to multiple workers, this fails with pickling error (probably ColumnSelector should be imported from separate file for that)
#  
#  
#  Please leave any feedback or improvement ideas!

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle

# Pipeline
from sklearn.pipeline import Pipeline, FeatureUnion
# Feature extraction
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
# Model
from sklearn.linear_model import LogisticRegression
# Cross-val
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from itertools import combinations, chain

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Your Alice data path
DATA_PATH = '../input'

# load train & kaggle test
train_data = pd.read_csv(DATA_PATH + '/train_sessions.csv')
submit_data = pd.read_csv(DATA_PATH + '/test_sessions.csv')

# frequently used column names
site_cols = ['site%d' % i for i in range(1, 11)]
time_cols = ['time%d' % i for i in range(1, 11)]

# light preprocessing
def convert_types(df):
    df[site_cols] = df[site_cols].fillna(0).astype('int')
    df[time_cols] = df[time_cols].apply(pd.to_datetime)
    return df

train_data = convert_types(train_data)
submit_data = convert_types(submit_data)

# sort by session start date, for time-based cross-validation
train_data = train_data.sort_values(by=time_cols[0])

# get the target column
y = train_data['target']

# Load websites dictionary
with open(DATA_PATH + '/site_dic.pkl', 'rb') as input_file:
    site_dict = pickle.load(input_file)

site_ids = list(site_dict.values())
# Create dataframe for the dictionary
sites_dict = pd.DataFrame(list(site_dict.keys()), index=list(site_dict.values()), columns=['site'])


# In[ ]:


#####################################
## Helper functions that extract different data
#####################################

# Return sites columns as a single string
# This string can be supplied into CountVectorizer or TfidfVectorizer
def extract_sites_as_string(X):
    return X[site_cols].astype('str').apply(' '.join, axis=1)

# Year-month feature from A4
def feature_year_month(X):
    return pd.DataFrame(X['time1'].dt.year * 100 + X['time1'].dt.month)

# Hour feature from A4
def feature_hour(X):
    return pd.DataFrame(X['time1'].dt.hour)

# Month
def feature_month(X):
    return pd.DataFrame(X['time1'].dt.month)

# Weekday
def feature_weekday(X):
    return pd.DataFrame(X['time1'].dt.weekday)

# Is morning feature from A4
def feature_is_morning(X):
    return pd.DataFrame(X['time1'].dt.hour <= 11)

# Session length feature from A4
def feature_session_len(X):
    X['session_end_time'] = X[time_cols].max(axis=1)
    X['session_duration'] = (X['session_end_time'] - X['time1']).astype('timedelta64[s]')
    return X[['session_duration']]


# Add more functions here :)
# ...


# In[ ]:


# Special transformer to save output shape
class ShapeSaver(BaseEstimator, TransformerMixin):
    def transform(self, X):
        self.shape = X.shape
        return X

    def fit(self, X, y=None, **fit_params):
        return self

###################################
## Defining the processing pipeline
## NOTE: ShapeSaver() is required as the last step for each feature
##################################
transform_pipeline = Pipeline([
    ('features', FeatureUnion([
        # List of features goes here:
        ('year_month_val', Pipeline([
            ('extract', FunctionTransformer(feature_year_month, validate=False)),
            ('scale', StandardScaler()),
            ('shape', ShapeSaver())
        ])),
        ('session_len', Pipeline([
            ('extract', FunctionTransformer(feature_session_len, validate=False)),
            ('scale', StandardScaler()),
            ('shape', ShapeSaver())
        ])),
        ('weekday_cat', Pipeline([
            ('extract', FunctionTransformer(feature_weekday, validate=False)),
            ('ohe', OneHotEncoder()),
            ('shape', ShapeSaver())
        ])),
        ('hour_val', Pipeline([
            ('extract', FunctionTransformer(feature_hour, validate=False)),
            ('scale', StandardScaler()),
            ('shape', ShapeSaver())
         ])),
        ('hour_cat', Pipeline([
            ('extract', FunctionTransformer(feature_hour, validate=False)),
            ('ohe', OneHotEncoder()),
            ('shape', ShapeSaver())
         ])),
        ('month_cat', Pipeline([
            ('extract', FunctionTransformer(feature_month, validate=False)),
            ('ohe', OneHotEncoder()),
            ('shape', ShapeSaver())
         ])),
        ('is_morning', Pipeline([
            ('extract', FunctionTransformer(feature_is_morning, validate=False)),
            ('shape', ShapeSaver())
         ])),
        ('sites_tfidf', Pipeline([
            ('extract', FunctionTransformer(extract_sites_as_string, validate=False)),
            ('count', TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')),
            ('shape', ShapeSaver())
        ])),
        # Add more features here :)
        # ...
    ]))
])

# Join train & submit data
full_df = pd.concat([train_data, submit_data])

# Remember train dataset size to split it later
train_size = train_data.shape[0]

# Run preprocessing on full data
full_transformed_df = transform_pipeline.fit_transform(full_df)


# In[ ]:


##########################################
## Define feature selection & model pipeline
##########################################

# Column selection transformer
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, ranges=[]):
        self.ranges = ranges
    
    def transform(self, X):
        # flatten ranges to plain list of column numbers
        cols = [i for r in self.ranges for i in r[1]]
        return X[:, cols]
    
    def fit(self, X, y=None, **fit_params):
        return self

# Pipeline to select features and fit model
pipeline = Pipeline([
    ('select_features', ColumnSelector()),
    # Our basic model
    ('model', LogisticRegression())
])


# In[ ]:


# Let's get list of all features and their columns from preprocessing Pipeline
features = [i[0] for i in transform_pipeline.get_params()['features'].get_params()['transformer_list']]
sizes = [transform_pipeline.get_params()['features'].get_params()[i + '__shape'].shape[1] for i in features]

# Create list of column ranges for each feature
feature_col_ranges = []
idx = 0
for size in sizes:
    feature_col_ranges.append(range(idx, size + idx))
    idx = size + idx

# Dict feature name => column range
feature_ranges = dict(zip(features, feature_col_ranges))

#####################################
## Features and hyper-params search!
####################################

# Combinations of features you want to check
# Names should match to step names defined in the processing pipeline
# Here's example of how more features does not result in better model; of these two the smaller set of features will yield better cross-val score
sel_features = [
    ['year_month_val', 'hour_cat', 'hour_val', 'session_len', 'weekday_cat', 'month_cat', 'is_morning', 'sites_tfidf'],
    ['hour_cat', 'session_len','month_cat', 'is_morning', 'weekday_cat', 'sites_tfidf'],
]

# Generate all possible combinations of features automatically, if you have fast machine to test them all
# Range defines min/max features number to try
# sel_features = list(chain.from_iterable(combinations(features, r) for r in range(4, 7)))

# Get ranges of columns for selected features
sel_col_ranges = list(map(lambda feats: [(f, feature_ranges[f]) for f in feats], sel_features))
param_grid = {
    # Feature selection params
    'select_features__ranges': sel_col_ranges,
    # Model hyper parameters, can try tuning these too
    'model__C': [1], #np.linspace(0.01, 5, 15)
    'model__random_state': [17]
}

# Double check param grid
print(param_grid)


# In[ ]:


# Grid search
# Unfortunately doesn't work with n_jobs=-1, let me know if you know how to fix this in kaggle kernel :)
grid = GridSearchCV(pipeline, param_grid=param_grid, scoring='roc_auc', cv=TimeSeriesSplit(n_splits=5), n_jobs=1, verbose=1)
# Take only train part of the full dataset
grid.fit(full_transformed_df[:train_size, :], y)

print('Best cross-val score: ', grid.best_score_)
print('Best params: ', grid.best_params_)


# In[ ]:


# Make predictions for submission data (take only submission data from full dataset)
submission = grid.predict_proba(full_transformed_df[train_size:, :])[:,1]

# Save to CSV
df = pd.DataFrame(submission, index=np.arange(1, submit_data.shape[0] + 1), columns=['target'])
df.to_csv('submission.csv', index_label='session_id')

