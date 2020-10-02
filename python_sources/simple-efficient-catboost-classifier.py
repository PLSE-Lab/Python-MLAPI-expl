#!/usr/bin/env python
# coding: utf-8

# # Gradient boosting decision trees are using here
# 
# Upvote if you like Gradient boosting decision trees.
# 
# * CatBoost library allows you to work with text data and preprocess it. 
# * And it works rather fast. 
# * CatBoost is perfect for handling with categorical data.
# 
# In my notebook I select hyperparameters. Fork right now and try your best in searching for params ([grid search](https://catboost.ai/docs/concepts/python-reference_catboost_grid_search.html) may help you).

# In[ ]:


import os
import gc
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

import catboost
print(catboost.__version__)
from catboost import CatBoostClassifier, Pool


# In[ ]:


# Loading data

train1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")[['comment_text', 'toxic']]
train1['lang'] = 'en'

lang = ['es', 'fr', 'pt', 'ru', 'it', 'tr']
for lan in lang:
    train_ = pd.read_csv(f'/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-{lan}-cleaned.csv')
    train_['lang'] = lan
    train1 = train1.append(train_[['comment_text', 'lang', 'toxic']], ignore_index=True)

train = train1.sample(n=300000).reset_index(drop=True)
del train1 
gc.collect()


# In[ ]:


test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
test.rename(columns={'content': 'comment_text'}, inplace=True)

subm = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')

val = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')[['comment_text', 'lang', 'toxic']]


# In[ ]:


train['comment_text'].fillna("", inplace=True)
test['comment_text'].fillna("", inplace=True)
val['comment_text'].fillna("", inplace=True)


# In[ ]:


# Creating pool objects

train_pool = Pool(
    train.drop(columns='toxic'), label=train['toxic'], 
    cat_features=['lang'],
    text_features=['comment_text'],
)

validation_pool = Pool(
    val.drop(columns='toxic'), label=val['toxic'], 
    cat_features=['lang'],
    text_features=['comment_text'],
)

test_pool = Pool(
    test,
    cat_features=['lang'],
    text_features=['comment_text'],
)


# In[ ]:


# Setting the params

clf_params = {
     'learning_rate': 0.07688199728727341,
     'depth': 6,
     'num_trees': 2000,
     'random_strength': 3,
     'bagging_temperature': 1.02,
     'eval_metric': 'AUC',
     'random_seed': 42,
    #  'logging_level': 'Silent',
     'task_type': 'GPU',
     'grow_policy': 'Lossguide',
     'text_processing': {
        "tokenizers" : [{
            'tokenizer_id': 'Space',
            'delimiter': ' ',
            'separator_type': 'ByDelimiter',
        },{
            'tokenizer_id': 'Sense',
            'separator_type': 'BySense',
        }],
        
        "dictionaries" : [{
            'dictionary_id': 'Word',
            'max_dictionary_size': '50000',
            "occurrence_lower_bound" : "3",
            'gram_order': '1',
        },{
            'dictionary_id': 'BiGram',
            'max_dictionary_size': '50000',
            "occurrence_lower_bound" : "3",
            'gram_order': '2',
        },{
            'dictionary_id': 'TriGram',
            'token_level_type': 'Letter',
            'max_dictionary_size': '50000',
            "occurrence_lower_bound" : "3",
            'gram_order': '3',
        }],

        "feature_processing" : {
            "default" : [{
                "dictionaries_names" : ["Word", "BiGram", "TriGram"],
                "feature_calcers" : ["BoW", "NaiveBayes", "BM25"],
                "tokenizers_names" : ["Space", "Sense"]
            },{
                "dictionaries_names" : ["Word", "BiGram"],
                "feature_calcers" : ["BoW", "BM25"],
                "tokenizers_names" : ["Sense"]
            },{
                "dictionaries_names" : ["Word", "TriGram"],
                "feature_calcers" : ["NaiveBayes", "BM25"],
                "tokenizers_names" : ["Sense"]
            },{
                "dictionaries_names" : ["Word"],
                "feature_calcers" : ["BoW"],
                "tokenizers_names" : ["Space"]
            }],
        }
    }
}


# In[ ]:


# Fitting

m = CatBoostClassifier(**clf_params)
m.fit(train_pool, eval_set=validation_pool, verbose=100, plot=True)


# In[ ]:


# Predicting

preds = m.predict_proba(test_pool)[:, 1]


# In[ ]:


submission = pd.DataFrame({'id': subm['id'].values, 'toxic': preds})
submission.to_csv('submission.csv', index=False)
submission

