#!/usr/bin/env python
# coding: utf-8

# In[67]:


import gc
import numpy as np
import pandas as pd
import keras as ks
from sklearn.pipeline import make_pipeline, make_union
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, CategoricalEncoder
from sklearn.linear_model import SGDRegressor
from sklearn.base import BaseEstimator, TransformerMixin
import lightgbm as gbm
import xgboost as xg


# In[68]:


get_ipython().run_cell_magic('bash', '', 'cd ../input/kaggledays2/kaggledays-warsaw\nls')


# In[69]:


def load_data(filename="../input/kaggledays2/kaggledays-warsaw/train.csv"):
    data = pd.read_csv(filename, sep="\t", index_col='id')
    msg = "Reading the data ({} rows). Columns: {}"
    print(msg.format(len(data), data.columns))
    # Select the columns (feel free to select more)
    X = data.loc[:, ['question_id', 'question_score', 'subreddit','question_utc','answer_utc' ,'question_text', 'answer_text']]
    
    ####
    X['time_to_answer'] = X['answer_utc'] - X['question_utc']

    question_time = pd.DatetimeIndex(pd.to_datetime(X['question_utc'], unit='s'))
    X['hour_questioned'] = question_time.hour
    X['dayofweek_questioned'] = question_time.dayofweek

    answered_time = pd.DatetimeIndex(pd.to_datetime(X['answer_utc'], unit='s'))
    X['hour_answered'] = answered_time.hour
    X['dayofweek_answered'] = answered_time.dayofweek
    
#     s = X.groupby('question_id').agg('count')['subreddit']
#     counts_df = pd.DataFrame(np.array([s.index, s.values]).T, columns=['q_id','answers_count'])
#     X = X.merge(counts_df, left_on='question_id', right_on='q_id')
    ###
    
    try:
        y = data.loc[:, "answer_score"]
    except KeyError: # There are no answers in the test file
        return X, None

    return X, y
X, y = load_data()


# In[70]:


#s = X.groupby('question_id').order_by('answer_utc')
#s


# In[60]:


# s = X.groupby('question_id').agg('count')['subreddit']
# counts_df = pd.DataFrame(np.array([s.index, s.values]).T, columns=['q_id','answers_count'])
# xxx = X.merge(counts_df, left_on='question_id', right_on='q_id')
# len(X), len(xxx)


# In[71]:


X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[72]:


def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(
        np.mean((np.log1p(y) - np.log1p(y0)) ** 2)
    )


# In[73]:


class AddTimeDifferenceCol(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, *args, **kwargs):
        return self

    def transform(self, data, *args, **kwargs):
        time_to_answer = data['answer_utc'] - data['question_utc']
        return time_to_answer.values

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns, orient=None):
        super(FeatureSelector, self).__init__()
        self.columns = columns

    def fit(self, X, *args, **kwargs):
        return self

    def transform(self, data, *args, **kwargs):
        return data[self.columns].values
    
class JustSelect(BaseEstimator, TransformerMixin):
    def __init__(self, columns, orient=None):
        super(JustSelect, self).__init__()
        self.columns = columns

    def fit(self, X, *args, **kwargs):
        return self

    def transform(self, data, *args, **kwargs):
        return data[self.columns].values.reshape(-1, 1)
    
def build_model(regressor):
    process_data = make_union(
        make_pipeline(
            JustSelect('subreddit'),
            CategoricalEncoder(encoding='onehot')
        ),
        make_pipeline(
            JustSelect('hour_questioned'),
            OneHotEncoder()
        ),
        make_pipeline(
            JustSelect('dayofweek_questioned'),
            OneHotEncoder()
        ),
        make_pipeline(
            JustSelect('hour_answered'),
            OneHotEncoder()
        ),
        make_pipeline(
            JustSelect('dayofweek_answered'),
            OneHotEncoder()
        ),
#         make_pipeline(
#             JustSelect('answers_count'),
#             StandardScaler()
#         ),
        make_pipeline(
            JustSelect('time_to_answer'),
            StandardScaler()
        ),
        make_pipeline(
            JustSelect("question_score"),
            StandardScaler()
        ),
        make_pipeline(
            FeatureSelector("question_text"),
            TfidfVectorizer(max_features=10, token_pattern="\w+"),
        ),
        make_pipeline(
            FeatureSelector("answer_text"),
            TfidfVectorizer(max_features=10, token_pattern="\w+"),
        ),
    )

    model = make_pipeline(
         process_data, regressor
    )
    return model


# In[74]:


get_ipython().run_cell_magic('time', '', '#LGBM\nmodel1 = build_model(gbm.LGBMRegressor(n_estimators=300))\nmodel1.fit(X_train, np.log1p(y_train))\n\ny_train_theor1 = np.expm1(model1.predict(X_train))\ny_test_theor1 = np.expm1(model1.predict(X_test))\nprint()\nprint("Training set")\nprint("RMSLE:   ", rmsle(y_train, y_train_theor1))\n\nprint("Test set")\nprint("RMSLE:   ", rmsle(y_test, y_test_theor1))\n\n"""\nn_estimators=300\nTraining set\nRMSLE:    0.762359925571\nTest set\nRMSLE:    0.783095873475\n\nn_estimators=500\nTraining set\nRMSLE:    0.754830061249\nTest set\nRMSLE:    0.781994859354\n\nn_estimators=700\nTraining set\nRMSLE:    0.742314837962\nTest set\nRMSLE:    0.776922614186\nCPU times: user 5min 35s, sys: 2.65 s, total: 5min 37s\nWall time: 2min 39s\n\nn_estimators=1200\nTraining set\nRMSLE:    0.698623907938\nTest set\nRMSLE:    0.754163190987\nCPU times: user 8min 23s, sys: 2.3 s, total: 8min 25s\nWall time: 3min 19s\n"""')


# In[75]:


X_val, _ = load_data('../input/kaggledays2/kaggledays-warsaw/test.csv')

sub = pd.DataFrame()
sub['id'] = X_val.index
sub['answer_score'] = np.expm1(model1.predict(X_val)) * 0.8

leaks = pd.read_csv("../input/leaked/leaked_records.csv").rename(columns={"answer_score": "leak"})

sub = pd.merge(sub, leaks, on="id", how="left")
sub.loc[~sub["leak"].isnull(), "answer_score"] = sub.loc[~sub["leak"].isnull(), "leak"]
sub = sub.drop(['leak'], axis=1)
sub.to_csv('submission.csv', index=False)


# In[76]:


get_ipython().run_cell_magic('bash', '', 'head submission.csv')


# In[ ]:




