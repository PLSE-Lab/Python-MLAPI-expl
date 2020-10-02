#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline, make_union
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.base import BaseEstimator, TransformerMixin


# In[5]:


def load_data(filename="../input/kaggledays-warsaw/train.csv"):
    data = pd.read_csv(filename, sep="\t", index_col='id')
    msg = "Reading the data ({} rows). Columns: {}"
    print(msg.format(len(data), data.columns))
    # Select the columns (feel free to select more)
    X = data.loc[:, ['question_text', 'answer_text']]
    try:
        y = data.loc[:, "answer_score"]
    except KeyError: # There are no answers in the test file
        return X, None
    return X, y
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[6]:


def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(
        np.mean((np.log1p(y) - np.log1p(y0)) ** 2)
    )


# In[ ]:


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns, orient=None):
        super(FeatureSelector, self).__init__()
        self.columns = columns

    def fit(self, X, *args, **kwargs):
        return self

    def transform(self, data, *args, **kwargs):
        return data[self.columns].values

def build_model():
    process_data = make_union(
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
         process_data,
         SGDRegressor(),
    )
    return model


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model = build_model()\nmodel.fit(X_train, np.log1p(y_train))\n\ny_train_theor = np.expm1(model.predict(X_train))\ny_test_theor = np.expm1(model.predict(X_test))\nprint()\nprint("Training set")\nprint("RMSLE:   ", rmsle(y_train, y_train_theor))\n\nprint("Test set")\nprint("RMSLE:   ", rmsle(y_test, y_test_theor))')


# In[ ]:


X_val, _ = load_data('../input/kaggledays-warsaw/test.csv')
solution = pd.DataFrame(index=X_val.index)
solution['answer_score'] = np.expm1(model.predict(X_val))
solution.to_csv('submission.csv')


# In[1]:





# In[ ]:




