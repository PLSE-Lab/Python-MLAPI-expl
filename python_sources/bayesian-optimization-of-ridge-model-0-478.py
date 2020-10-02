#!/usr/bin/env python
# coding: utf-8

# # Hyper Parameter Tuning With Bayesian Optimization
# 
# Bayesian Optimization is a powerful way to tune hyper parameters in a model. Instead of a brute force approach like with sklearn's grid_search, it tries to optimize the loss function by intelligently exploring the underlying distribution.  This script can be run with the develop flag to find the optimal hyperparameters, then to deploy a final model, specify the optimal found parameters by hand. 
# 
# To see more of the documentation, check out the project github:
# https://github.com/fmfn/BayesianOptimization

# In[ ]:


import gc
import time
import numpy as np
import pandas as pd
import sys
from scipy.sparse import csr_matrix, hstack

import warnings
warnings.filterwarnings('ignore')

start_time = time.time()
tcurrent   = start_time

np.random.seed(54)   

NUM_BRANDS = 4550
NUM_CATEGORIES = 1280
MAX_FEATURES_NAME = 10000
MAX_FEATURES_DESC =10000

develop = True


# In[ ]:


from sklearn.metrics import make_scorer

def rmsle(y_true, y_pred):
    # Remember, we transformed price with log1p previously.
    return np.sqrt(mean_squared_log_error(np.expm1(y_true), np.expm1(y_pred)))

neg_rmsle = make_scorer(rmsle, greater_is_better=False)

def split_cat(text):
    try:
        return text.split("/")
    except:
        return ("No Label", "No Label", "No Label")


def handle_missing_inplace(dataset):
    dataset['general_cat'].fillna(value='missing', inplace=True)
    dataset['subcat_1'].fillna(value='missing', inplace=True)
    dataset['subcat_2'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)


def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category1 = dataset['general_cat'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category2 = dataset['subcat_1'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category3 = dataset['subcat_2'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    dataset.loc[~dataset['general_cat'].isin(pop_category1), 'general_cat'] = 'missing'
    dataset.loc[~dataset['subcat_1'].isin(pop_category2), 'subcat_1'] = 'missing'
    dataset.loc[~dataset['subcat_2'].isin(pop_category3), 'subcat_2'] = 'missing'


def to_categorical(dataset):
    dataset['general_cat'] = dataset['general_cat'].astype('category')
    dataset['subcat_1'] = dataset['subcat_1'].astype('category')
    dataset['subcat_2'] = dataset['subcat_2'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')


# In[ ]:


from time import gmtime, strftime
print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

train = pd.read_table('../input/train.tsv', engine='c')
test = pd.read_table('../input/test.tsv', engine='c')


train = train.drop(train[(train.price == 0.0)].index)
y = np.log1p(train["price"])

print('[{}] Finished to load data'.format(time.time() - start_time))
print('Train shape: ', train.shape)
print('Test shape: ', test.shape)

submission: pd.DataFrame = test[['test_id']]
    
train.head()


# In[ ]:


import re, string, timeit

regex = re.compile('[%s]' % re.escape(string.punctuation))

train['item_description'] = train['item_description'].apply(lambda x: regex.sub('',str(x).lower()))
test['item_description'] = test['item_description'].apply(lambda x: regex.sub('',str(x).lower()))
train['name'] = train['name'].apply(lambda x: regex.sub('',x.lower()))
test['name'] = test['name'].apply(lambda x: regex.sub('',x.lower()))

train.head()


# In[ ]:


train['general_cat'], train['subcat_1'], train['subcat_2'] =         zip(*train['category_name'].apply(lambda x: split_cat(x)))
train.drop('category_name', axis=1, inplace=True)

test['general_cat'], test['subcat_1'], test['subcat_2'] =         zip(*test['category_name'].apply(lambda x: split_cat(x)))
test.drop('category_name', axis=1, inplace=True)
print('[{}] Split categories completed.'.format(time.time() - start_time))


handle_missing_inplace(train)
handle_missing_inplace(test)
print('[{}] Handle missing completed.'.format(time.time() - start_time))


cutting(train)
cutting(test)
print('[{}] Cut completed.'.format(time.time() - start_time))


to_categorical(train)
to_categorical(test)
print('[{}] Convert categorical completed'.format(time.time() - start_time))
train.head()


# In[ ]:


from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

pipeline = Pipeline([
            ('selector', ItemSelector(key='name')),
            ('tfidf', TfidfVectorizer(ngram_range = (1, 3),
                            strip_accents = 'unicode', 
                            stop_words = 'english',
                            min_df=20,
                            max_df=.9,
                            max_features = MAX_FEATURES_NAME))
])

X_name = pipeline.fit_transform(train, train['price'])
X_name_test = pipeline.transform(test)

X_name.shape


# In[ ]:


pipeline = Pipeline([
    ('item_description', Pipeline([
            ('selector', ItemSelector(key='item_description')),
            ('tfidf', TfidfVectorizer(ngram_range = (1, 3),
                                stop_words = 'english',
                                min_df=20,
                                max_df=.9,
                                max_features = MAX_FEATURES_DESC))
            ]))
])

X_text = pipeline.fit_transform(train, train['price'])
X_text_test = pipeline.transform(test)

X_text.shape


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer

wb = CountVectorizer()
X_category1 = wb.fit_transform(train['general_cat'])
X_category1_test = wb.transform(test['general_cat'])
X_category2 = wb.fit_transform(train['subcat_1'])
X_category2_test = wb.transform(test['subcat_1'])
X_category3 = wb.fit_transform(train['subcat_2'])
X_category3_test = wb.transform(test['subcat_2'])
print('[{}] Count vectorize `categories` completed.'.format(time.time() - start_time))


lb = LabelBinarizer(sparse_output=True)
X_brand = lb.fit_transform(train['brand_name'])
X_brand_test = lb.transform(test['brand_name'])
print('[{}] Label binarize `brand_name` completed.'.format(time.time() - start_time))


# In[ ]:


X_dummies = csr_matrix(pd.get_dummies(train[['item_condition_id', 'shipping']],
                                          sparse=True).values)
X_dummies_test = csr_matrix(pd.get_dummies(test[['item_condition_id', 'shipping']],
                                          sparse=True).values)
print('[{}] Get dummies on `item_condition_id` and `shipping` completed.'.format(time.time() - start_time))
print(X_dummies.shape, X_brand.shape, X_category1.shape, X_category2.shape, X_category3.shape,
          X_text.shape)


sparse_merge = hstack((X_dummies, X_brand, X_name, X_text,
                       X_category1, X_category2, X_category3)).tocsr()
sparse_merge_test = hstack((X_dummies_test, X_brand_test, X_name_test, X_text_test,
                            X_category1_test, X_category2_test, X_category3_test)).tocsr()

print('[{}] Create sparse merge completed'.format(time.time() - start_time))
del X_dummies, lb, X_brand, X_category1, X_category2, X_category3
del X_name, X_text
del X_dummies_test, X_brand_test, X_category1_test, X_category2_test, X_category3_test
del X_name_test, X_text_test

gc.collect()


# In[ ]:


from sklearn.model_selection import train_test_split

X = sparse_merge
X_test = sparse_merge_test

train_X, train_y = X, y
del X, sparse_merge, sparse_merge_test
gc.collect()


# In[ ]:


from bayes_opt import BayesianOptimization
from sklearn.metrics import make_scorer, mean_squared_log_error
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Ridge

seed = 101 # Lucky seed

    
def target(**params):
    fit_intercept = int(params['fit_intercept'])
    fit_intercept_dict = {0:False, 1:True}

    model = Ridge(alpha = params['alpha'],
                    fit_intercept = fit_intercept_dict[fit_intercept],
                    copy_X = True)
    
    scores = cross_val_score(model, train_X, train_y, scoring=neg_rmsle, cv=3)
    return scores.mean()
    
params = {'alpha':(1, 4),
          'fit_intercept':(0,1.99)}
if develop:
    bo = BayesianOptimization(target, params, random_state=seed)
    bo.gp.set_params(alpha=1e-8)
    bo.maximize(init_points=5, n_iter=10, acq='ucb', kappa=2)
    
    print(bo.res['max']['max_params'])


# In[ ]:


model = Ridge(alpha = 3.0656,
                  fit_intercept = True,
                  copy_X = True)

model.fit(train_X, train_y)
predsR = model.predict(X_test)
submission['price'] = np.expm1(predsR)
submission.to_csv("Bayesian_Ridge.csv", index=False)


# In[ ]:


nm=(time.time() - start_time)/60
print ("Total processing time %s min" % nm)

