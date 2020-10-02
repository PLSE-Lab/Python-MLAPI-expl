#!/usr/bin/env python
# coding: utf-8

# A relatively decent score for my very first competition. I tried doing an ensemble of simple models but couldn't manage to get better results than with the single LGB with a large number of leaves and long training. Definitely something to learn.
# 
# The whole thing takes ~59min, with ~10 of prediction and ~2 of preprocessing.
# 
# **Comments:**
# - I join the columns with freely-entered text into one. Treating them separately did not improve the results.
# - Label encoding of brands and categories is done manually as a function of mean price. This worked better for me than an unsorted encoding. Unseen data is handled as missing the brand/category name (perhaps the median would have been better?).
# - Missing values in `shipping`, if any, are treated as if the shipping is paid by the buyer.
# - I made the idiotic mistake of encoding missing values in `item_condition` with a 0 and then applying OHE. There were no missing entries either in the training set or reduced testing set, but if there are any in the full testing set then the kernel will fail because the model was trained with 5 columns per item condition instead of 6.
# - In case there are no missing values for `item_condition` in the full testing set and the kernel survives the time and memory constraints, the finall score may be affected by the fact that I filled missing values of freely-entered text AFTER combining the two fields, so eg. properly named listings with a null description will be blank.

# In[ ]:


import numpy as np
import pandas as pd

import scipy

import gc

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split

import lightgbm as lgb


# In[ ]:


develop = False


# In[ ]:


add_stop = ['[rm]', 'rm']
stop_words = ENGLISH_STOP_WORDS.union(add_stop)

tfidf = TfidfVectorizer(stop_words=stop_words, max_features=50000)


# In[ ]:


def preprocess(dataset, train=True, brands_by_price=None, cat1_by_price=None, cat2_by_price=None, cat3_by_price=None):
    
    dataset.replace('No description yet', '', inplace=True)
    dataset['text'] = dataset['name'] + ' ' + dataset['item_description']
    dataset['text'].fillna('', inplace=True)
    
    if train:
        tfidfvec = tfidf.fit_transform(dataset['text'])
    else:
        tfidfvec = tfidf.transform(dataset['text'])
        
    dataset['brand_name'].fillna('missing_brand', inplace=True)
    
    dataset['item_condition_id'].fillna(0, inplace=True)

    dataset['shipping'].fillna(0, inplace=True)

    dataset['category_name'].fillna('missing_cat', inplace=True)

    dataset['main_cat'] = dataset['category_name'].apply(lambda name: name.split('/')[0])
    dataset['second_cat'] = dataset['category_name'].apply(lambda name: name.split('/')[1] if len(name.split('/')) > 1 else 'missing_cat')
    dataset['third_cat'] = dataset['category_name'].apply(lambda name: name.split('/')[2] if len(name.split('/')) > 2 else 'missing_cat')
    
    if train:
        brands_by_price = dataset.groupby('brand_name').mean()['price'].sort_values(ascending=False).to_frame()
        brands_by_price['id'] = brands_by_price.reset_index().index.values
        n_brands = len(brands_by_price)
        brand_names = brands_by_price.index.values
        
        cat1_by_price = dataset.groupby('main_cat').mean()['price'].sort_values(ascending=False).to_frame()
        cat1_by_price['id'] = cat1_by_price.reset_index().index.values
        n_cat1 = len(cat1_by_price)
        cat1_names = cat1_by_price.index.values
        
        cat2_by_price = dataset.groupby('second_cat').mean()['price'].sort_values(ascending=False).to_frame()
        cat2_by_price['id'] = cat2_by_price.reset_index().index.values
        n_cat2 = len(cat2_by_price)
        cat2_names = cat2_by_price.index.values
        
        cat3_by_price = dataset.groupby('third_cat').mean()['price'].sort_values(ascending=False).to_frame()
        cat3_by_price['id'] = cat3_by_price.reset_index().index.values
        n_cat3 = len(cat3_by_price)
        cat3_names = cat3_by_price.index.values
        
        dataset=dataset.drop(['price'], axis=1)
    
    else:
        n_brands = len(brands_by_price)
        brand_names = brands_by_price.index.values
        dataset['brand_name'] = dataset['brand_name'].apply(lambda name: name if name in brand_names else 'missing_brand')
        
        n_cat1 = len(cat1_by_price)
        cat1_names = cat1_by_price.index.values
        dataset['main_cat'] = dataset['main_cat'].apply(lambda name: name if name in cat1_names else 'missing_cat')
        
        n_cat2 = len(cat2_by_price)
        cat2_names = cat2_by_price.index.values
        dataset['second_cat'] = dataset['second_cat'].apply(lambda name: name if name in cat2_names else 'missing_cat')
        
        n_cat3 = len(cat3_by_price)
        cat3_names = cat3_by_price.index.values
        dataset['third_cat'] = dataset['third_cat'].apply(lambda name: name if name in cat3_names else 'missing_cat')
        
    
    brand_data = brands_by_price.loc[dataset['brand_name']]
    dataset['brand_id'] = brand_data['id'].values/n_brands
    
    cat1_data = cat1_by_price.loc[dataset['main_cat']]
    dataset['cat1_id'] = cat1_data['id'].values/n_cat1
    
    cat2_data = cat2_by_price.loc[dataset['second_cat']]
    dataset['cat2_id'] = cat2_data['id'].values/n_cat2
    
    cat3_data = cat3_by_price.loc[dataset['third_cat']]
    dataset['cat3_id'] = cat3_data['id'].values/n_cat3
    
    ohe_data = pd.concat([pd.get_dummies(dataset[col], prefix=col) for col in dataset[['item_condition_id', 'shipping']]], axis=1)
    
    dataset = dataset.drop(['item_description', 'name', 'category_name', 'brand_name', 'main_cat', 'second_cat', 'third_cat', 'text', 'item_condition_id', 'shipping'], axis=1)
    feat_data = scipy.sparse.hstack((dataset.values, ohe_data.values, tfidfvec)).tocsr()
    
    del dataset
    gc.collect()
    
    if train:
        return feat_data, brands_by_price, cat1_by_price, cat2_by_price, cat3_by_price
    return feat_data


# In[ ]:


train_data = pd.read_csv('../input/train.tsv', sep='\t', index_col='train_id')

train_data = train_data[train_data.price >= 3.]

y=np.log1p(train_data.price.values)

feat_train, brands_by_price, cat1_by_price, cat2_by_price, cat3_by_price = preprocess(train_data)

del train_data
gc.collect()


# In[ ]:


params = {'num_leaves': 350, 'learning_rate': 0.1, 'feature_fraction': 0.9, 'bagging_fraction': 0.7, 'bagging_freq': 5, 'metric': 'l2_root',  'data_random_seed': 0, 'num_threads': 4, 'max_bin': 64}

if develop:
    X_train, X_test, y_train, y_test = train_test_split(feat_train, y, test_size=0.1, random_state=0)
    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_test, label=y_test, reference=train_set)
    
    bst = lgb.train(params, train_set, num_boost_round=900, valid_sets=[train_set, valid_set], early_stopping_rounds=100, verbose_eval=True)
    
else:
    train_set = lgb.Dataset(feat_train, label=y)
    
    bst = lgb.train(params, train_set, num_boost_round=900, valid_sets=[train_set], early_stopping_rounds=100, verbose_eval=True)


# In[ ]:


predictions = pd.DataFrame({'test_id': [], 'price': []})

test_chunks = pd.read_csv('../input/test.tsv', sep='\t', index_col='test_id', chunksize=350000)

for chunk in test_chunks:
    testId = chunk.index
    
    feat_test = preprocess(chunk, False, brands_by_price, cat1_by_price, cat2_by_price, cat3_by_price)
        
    preds = pd.DataFrame({'test_id': testId, 'price': np.expm1(bst.predict(feat_test, num_iteration=bst.best_iteration))})

    del feat_test
    gc.collect()

    predictions = pd.concat([predictions, preds], join="inner")

predictions.test_id = predictions.test_id.astype(int)
predictions.to_csv('submission.csv', index=False)

