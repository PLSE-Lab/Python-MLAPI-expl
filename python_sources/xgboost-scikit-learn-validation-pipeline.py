#!/usr/bin/env python
# coding: utf-8

# This kernel contain just some basic pipeline with scikit-learn "pipeline" building, cross-validation (with implemented contest metric) and some memory tricks.
# 
# # Environment preparation

# In[ ]:


import numpy as np

def reduce_mem_usage(df):
    """
    Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
    """
    for col in df.columns:
        col_type = df[col].dtype
        col_typename = str(col_type)
        is_numeric = col_typename.startswith('int') or             col_typename.startswith('float') or             col_typename.startswith('bool')
        if is_numeric:
            c_min = df[col].min()
            c_max = df[col].max()
            if col_typename.startswith('int') or col_typename.startswith('bool'):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    return df


# In[ ]:


from kaggle.competitions import twosigmanews
import gc

env = twosigmanews.make_env()
env._var07 = reduce_mem_usage(env._var07)
env._var10 = reduce_mem_usage(env._var10)
gc.collect()


# In my case, at this stage it used ~5.5Gb memory instead of 7-8Gb without reducing dataframes. 
# 
# # scikit-learn "utilities" functions

# In[ ]:


import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer

from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import FunctionTransformer, LabelBinarizer, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import TruncatedSVD
from xgboost import XGBClassifier


# ## Validation approach
# 
# I'll use validation based on time-series splitting of dates. So it'll work next way:
# - split dates to N ranges
# - when on each range:
#   - fit on dates before edge
#   - validate on later dates
# 
# But:
# - I need to predict sign of 10-day returns, not returns
# - in metric funtion I can pass only predictions and real values. 
# 
# And to calculate metric value I need more information then just true returns and predicted. I also need "universe" feature and dates, which can't be passed to scorer. So I'll make next thing:
# - create wrapper "estimator", which:
#   - fit wrapped model to predict returns sign
#   - after building prediction:
#     - calculate ```recordMetric = confidenceValue * returns * universe```
#     - then aggregate daily metric: ```dailyMetric = sum(recordMetric[day] for day in dates)```
#     - return such aggregated metric instead of prediction
# - then in scoring function I'll calculate next thing: ```metric = mean(dailyMetric) / std(dailyMetric)```
# 
# Finally, I'll wrap all this things usage in cross-validation function.
#   

# In[ ]:


class DateSplit(TimeSeriesSplit):
    def split(self, X, y, *args, **kwargs):
        assert isinstance(X, pd.DataFrame)
        dates = sorted(X['date'].unique())
        for train_idx, _ in super(self.__class__, self).split(dates):
            train_before = dates[train_idx[-1]]
            train_mask = (X['date'] <= train_before).values
            val_mask = (X['date'] > train_before).values
            yield np.where(train_mask)[0], np.where(val_mask)[0]
            
            
def metric_aggregation(_, daily_metrics):
    return daily_metrics.mean() / daily_metrics.std()


class DailyMetricEstimator(BaseEstimator):
    def __init__(self, base_model, pass_collumns=None):
        self.base_model = base_model
        self.pass_collumns = pass_collumns
        
    def _get_numeric_columns(self, df):
        columns = []
        for column in df.columns:
            typename = str(df[column].dtype)
            if typename.startswith('int') or typename.startswith('float') or typename.startswith('bool'):
                columns.append(column)
        return columns
                
    def _get_pass_collumns(self, X):
        if self.pass_collumns is not None:
            pass_collumns = self.pass_collumns
        else:
            pass_collumns = self._get_numeric_columns(X)
        return pass_collumns        
    
    def fit(self, X, y):
        assert isinstance(X, pd.DataFrame)
        pass_collumns = self._get_pass_collumns(X)
        y_classes = 1.0 * (y > 0.0)
        self.base_model.fit(X[pass_collumns], y_classes)
    
    def predict(self, X):
        assert isinstance(X, pd.DataFrame)
        X = X.copy(deep=False)
        pass_collumns = self._get_pass_collumns(X)
        X['confidenceValue'] = 2.0 * self.base_model.predict(X[pass_collumns]) - 1.0
        X['metric'] = X['returnsOpenNextMktres10'] * X['universe'] * X['confidenceValue']
        metric = X.groupby('date')['metric'].sum().values
        return metric


def twosigma_cross_val_score(model, X, y, pass_collumns=None):
    decorated_model = DailyMetricEstimator(model, pass_collumns)
    return cross_val_score(decorated_model, X, y, cv=DateSplit(), scoring=make_scorer(metric_aggregation))


# ## Pipeline-friendly LabelBinarizer
# 
# Standard```scikit-learn```'s ```LabelBinarizer``` gets fit-arguments related exception if you're trying to use it as feature extractor in ```Pipeline```. Here you could see customized class.

# In[ ]:


class LabelBinarizerPipelineFriendly(LabelBinarizer):
    def fit(self, X, y=None):
        """this would allow us to fit the model based on the X input."""
        super(LabelBinarizerPipelineFriendly, self).fit(X)
        
    def transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).transform(X)
    
    def fit_transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).fit(X).transform(X)


# ## repeater "Estimator"
# I'm using next class to build pipelines like
# ```
# Pipeline([
#     ('features', FeatureUnion([
#         ('feature1pipeline', Pipeline([
#             ...
#         ]),
#         ('feature2pipeline', Pipeline([
#             ...
#         ]),
#     ])),
#     ('clf', Predictor()),
# ])
# ```
# without it - building of ```feature1pipeline``` and ```feature2pipeline``` will be failed (seems like ```Pipeline``` must have extimator at last stage, not transformer).
# It could be usefull when we're going to apply different complex pipelines to different features.
# E.g. :
# - encode embeddings for categorical features
# - scale numeric feature

# In[ ]:


class Repeater(BaseEstimator):
    def fit(self, *args, **kwargs):
        return self
    
    def predict(self, X, *args, **kwargs):
        return X
    
    def transform(self, X, *args, **kwargs):
        return X
    
    def fit_transform(self, X, *args, **kwargs):
        return self.fit(X).transform(X)


# ## Building submission
# 
# Finally, next function will train model on your dataset and build submission. It's consuming next params:
# 
# - model
# - dataframe with features and target (see description below)
# - function that consume market data and news data and return such dataframe
# - list of collumns used by model

# In[ ]:


def make_submission(model, X, extract_features, pass_collumns=None):
    def get_numeric_columns(df):
        columns = []
        for column in df.columns:
            typename = str(df[column].dtype)
            if typename.startswith('int') or typename.startswith('float') or typename.startswith('bool'):
                columns.append(column)
        return columns
                
    if pass_collumns is None:
        pass_collumns = get_numeric_columns(X)
    
    y = 1.0 * (X['returnsOpenNextMktres10'] > 0.0)
    
    model.fit(X, y)
    for market_obs_df, news_obs_df, predictions_template_df in env.get_prediction_days():
        X_val = extract_features(market_obs_df, news_obs_df)
        confidence = model.predict(X_val) * 2.0 - 1.0
        prediction = pd.DataFrame({
            'assetCode': X_val['assetCode'],
            'confidenceValue': confidence,
        })
        env.predict(prediction)
    env.write_submission_file()


# # Extracting features
# 
# Below I'll define feature-extraction function (now it'll only use market data) and extract features for train set:

# In[ ]:


def extract_features(market_df, news_df):
    market_df['date'] = pd.to_datetime(market_df['time'].dt.date)
    news_df['date'] = pd.to_datetime(news_df['time'].dt.date)
    return reduce_mem_usage(market_df)


df = extract_features(*env.get_training_data())
df.head()


# # Dummy model
# 
# Firstly I'll check DummyClassifier to check cross-validation works:

# In[ ]:


twosigma_cross_val_score(DummyClassifier(), df, df['returnsOpenNextMktres10'])


# # XGBoost baseline
# 
# Next one is baseline with xgboost on market data. Note few things (at least, it was in my case):
# - training isn't fast enough, so maybe you'll need to work next way:
#   - cross-validate model
#   - then if yound score good enought - comment call for cross-validation and save score before commiting your work

# In[ ]:


market_numeric_features = ['volume', 'close', 'open', 'returnsClosePrevRaw1',
                           'returnsOpenPrevRaw1', 'returnsClosePrevMktres1',
                           'returnsOpenPrevMktres1', 'returnsClosePrevRaw10',
                           'returnsOpenPrevRaw10', 'returnsClosePrevMktres10',
                           'returnsOpenPrevMktres10']
pass_collumns = market_numeric_features + ['assetCode']
clf = Pipeline([
    ('features', FeatureUnion([
        ('assetCodeEncoder', Pipeline([
            ('selector', FunctionTransformer(lambda X: X['assetCode'], 
                                             validate=False)),
            ('label', LabelBinarizerPipelineFriendly(sparse_output=True)),
            ('t-svd', TruncatedSVD(10)),
            ('repeater', Repeater()),
        ])),
        ('numericFeatures', Pipeline([
            ('selector', FunctionTransformer(lambda X: X[market_numeric_features], 
                                             validate=False)),
            ('fillNa', FunctionTransformer(lambda X: X.fillna(0), 
                                           validate=False)),
            ('scale', StandardScaler()),
            ('repeater', Repeater()),
        ]))
    ])),
    ('classifier', XGBClassifier(n_estimators=50, tree_kind='hist'))
])


# In[ ]:


twosigma_cross_val_score(clf, df, df['returnsOpenNextMktres10'], pass_collumns=pass_collumns)


# # Building final submission

# In[ ]:


make_submission(clf, df, extract_features, pass_collumns)

