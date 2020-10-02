#!/usr/bin/env python
# coding: utf-8

# In[18]:


import eli5
from IPython.display import display
import numpy as np
import pandas as pd
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.base import BaseEstimator, TransformerMixin


# In[2]:


def load_data(filename="../input/kaggledays-warsaw/train.csv"):
    data = pd.read_csv(filename, sep="\t", index_col='id')
    msg = "Reading the data ({} rows). Columns: {}"
    print(msg.format(len(data), data.columns))
    try:
        y = data.loc[:, "answer_score"]
    except KeyError: # There are no answers in the test file
        return data, None
    return data, y

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[3]:


def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(
        np.mean((np.log1p(y) - np.log1p(y0)) ** 2)
    )


# In[4]:


X_train.head()


# In[35]:


get_ipython().run_cell_magic('time', '', '\nCOLUMNS = list(X_train.columns)\ndefault_preprocessor = TfidfVectorizer().build_preprocessor()\n\n\ndef field_extractor(field):\n    field_idx = COLUMNS.index(field)\n    return lambda x: default_preprocessor(x[field_idx])\n\n\nclass FeatureSelector(BaseEstimator, TransformerMixin):\n    def __init__(self, columns, fn=lambda x: x):\n        super().__init__()\n        self.columns = columns\n        self.field_idx = [COLUMNS.index(c) for c in columns]\n        self.fn = fn\n\n    def fit(self, X, *args, **kwargs):\n        return self\n\n    def transform(self, data, *args, **kwargs):\n        if isinstance(data, list):\n            data = np.array(data)\n        return self.fn(data[:, self.field_idx])\n    \n    def get_feature_names(self):\n        return self.columns\n\n    \nvectorizer = FeatureUnion([\n    # (\'q_score\', FeatureSelector([\'question_score\'], fn=lambda x: np.log1p(x.astype(int)))),\n    # (\'subreddit\', CountVectorizer(token_pattern=\'\\w+\', preprocessor=field_extractor(\'subreddit\'))),\n    (\'question\', TfidfVectorizer(max_features=10000, token_pattern="\\w+", preprocessor=field_extractor(\'question_text\'))),\n    (\'answer\', TfidfVectorizer(max_features=10000, token_pattern="\\w+", preprocessor=field_extractor(\'answer_text\'))),\n    ])\n\nmodel = SGDRegressor(max_iter=5)\nmodel.fit(vectorizer.fit_transform(X_train.values), np.log1p(y_train.values));\n\nprint("Valid RMSLE:", rmsle(y_test, np.expm1(model.predict(vectorizer.transform(X_test.values)))))')


# In[36]:


eli5.explain_weights(model, vectorizer, top=50)


# In[37]:


# eli5.explain_weights(model, vectorizer, top=100, feature_filter=lambda x: not x.startswith('subreddit_'))


# In[39]:


test_sample = X_test.sample(n=10, random_state=42)
for row in test_sample.values:
    display(eli5.show_prediction(model, row, vec=vectorizer))

