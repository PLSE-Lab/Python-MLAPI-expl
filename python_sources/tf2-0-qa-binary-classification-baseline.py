#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gc
import json
import numpy as np 
import pandas as pd
import re

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm_notebook as tqdm
import Levenshtein 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression

from scipy import spatial
import lightgbm as lgb
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


# In[ ]:


html_tags = ['<P>', '</P>', '<Table>', '</Table>', '<Tr>', '</Tr>', '<Ul>', '<Ol>', '<Dl>', '</Ul>', '</Ol>',              '</Dl>', '<Li>', '<Dd>', '<Dt>', '</Li>', '</Dd>', '</Dt>']
r_buf = ['is', 'are', 'do', 'does', 'did', 'was', 'were', 'will', 'can', 'the', 'a', 'an', 'of', 'in', 'and', 'on',          'what', 'where', 'when', 'which'] + html_tags

def clean(x):
    x = x.lower()
    for r in r_buf:
        x = x.replace(r, '')
    x = re.sub(' +', ' ', x)
    return x


# ## Prepairing train dataset
# 
# Here we are going to collect data from json files and format it to the tabular data. We will formulate the problem as a binary classification problem and will try to classify if chosen candidate is an answer.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'n_samples = 75000 # Number of samples to read from the train.json\n\n# Read data from train.json and prepare features\nids = []\nquestion_tfidfs = []\nanswer_tfidfs = []\ncandidates_str = []\ntargets = []\ntargets_str = []\ntargets_str_short = []\nfeatures = []\nrank_features = []\n\nwith open(\'/kaggle/input/tensorflow2-question-answering/simplified-nq-train.jsonl\', \'r\') as json_file:\n    cnt = 0\n    for line in tqdm(json_file):\n        json_data = json.loads(line) \n        \n        # TFIDF for document\n        stop_words = text.ENGLISH_STOP_WORDS.union(["book"])\n        tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words=stop_words)\n        tfidf.fit([json_data[\'document_text\']])\n\n        # TFIDF for question\n        question = json_data[\'question_text\']\n        question_tfidf = tfidf.transform([question]).todense()\n        \n        # Collect annotations\n        start_token_true = json_data[\'annotations\'][0][\'long_answer\'][\'start_token\']\n        end_token_true = json_data[\'annotations\'][0][\'long_answer\'][\'end_token\']\n        \n        # Collect short annotations\n        if json_data[\'annotations\'][0][\'yes_no_answer\'] == \'NONE\':\n            if len(json_data[\'annotations\'][0][\'short_answers\']) > 0:\n                s_ans = str(json_data[\'annotations\'][0][\'short_answers\'][0][\'start_token\']) + \':\' + \\\n                    str(json_data[\'annotations\'][0][\'short_answers\'][0][\'end_token\'])\n            else:\n                s_ans = \'\'\n        else:\n            s_ans = json_data[\'annotations\'][0][\'yes_no_answer\']\n\n        cos_d_buf = []\n        euc_d_buf = []\n        lev_d_buf = []\n        \n        doc_tokenized = json_data[\'document_text\'].split(\' \')\n        candidates = json_data[\'long_answer_candidates\']\n        candidates = [c for c in candidates if c[\'top_level\'] == True]\n        \n        if start_token_true != -1:\n            for c in candidates:\n                ids.append(str(json_data[\'example_id\']))\n\n                # TFIDF for candidate answer\n                start_token = c[\'start_token\']\n                end_token = c[\'end_token\']\n                answer = \' \'.join(doc_tokenized[start_token:end_token])\n                answer_tfidf = tfidf.transform([answer]).todense()\n\n                # Extract some features\n                cos_d = spatial.distance.cosine(question_tfidf, answer_tfidf)\n                euc_d = np.linalg.norm(question_tfidf - answer_tfidf)\n                lev_d = Levenshtein.distance(clean(question), clean(answer))\n                lev_r = Levenshtein.ratio(clean(question), clean(answer))\n                jar_s = Levenshtein.jaro(clean(question), clean(answer))\n                jaw_s = Levenshtein.jaro_winkler(clean(question), clean(answer))\n                tfidf_score = np.sum(question_tfidf*answer_tfidf.T)\n                question_tfidf_sum = np.sum(question_tfidf)\n                answer_tfidf_sum = np.sum(answer_tfidf)\n\n                features.append([\n                    cos_d, \n                    euc_d, \n                    lev_d, \n                    lev_r, \n                    jar_s, \n                    jaw_s, \n                    tfidf_score, \n                    question_tfidf_sum, \n                    answer_tfidf_sum\n                ])\n                \n                cos_d_buf.append(cos_d)\n                euc_d_buf.append(euc_d)\n                lev_d_buf.append(lev_d)\n\n                targets_str.append(str(start_token_true) + \':\' + str(end_token_true))\n                candidates_str.append(str(start_token) + \':\' + str(end_token))\n                targets_str_short.append(s_ans)\n\n                # Get target\n                if start_token == start_token_true and end_token == end_token_true:\n                    target = 1\n                else:\n                    target = 0\n                targets.append(target)\n\n            rank_cos_d = np.argsort(cos_d_buf)\n            rank_euc_d = np.argsort(euc_d_buf)\n            rank_lev_d = np.argsort(lev_d_buf)\n            rank_cos_d_ismin = (cos_d_buf == np.nanmin(cos_d_buf)).astype(int)\n            rank_euc_d_ismin = (euc_d_buf == np.nanmin(euc_d_buf)).astype(int)\n            rank_lev_d_ismin = (lev_d_buf == np.nanmin(lev_d_buf)).astype(int)\n            rank_features.append(np.array([rank_cos_d, rank_euc_d, rank_lev_d, \\\n                                           rank_cos_d_ismin, rank_euc_d_ismin, rank_lev_d_ismin]).T)\n\n        cnt += 1\n        if cnt >= n_samples:\n            break\n        \ntrain = pd.DataFrame()\ntrain[\'example_id\'] = ids\ntrain[\'target\'] = targets\ntrain[\'CorrectString\'] = targets_str\ntrain[\'CorrectString_short\'] = targets_str_short\ntrain[\'CandidateString\'] = candidates_str\n\nfeatures = np.array(features)\nfeatures_df = pd.DataFrame(features)\nfeatures_df.columns = [f\'feature_{i}\' for i in range(features.shape[1])]\ntrain = pd.concat([train, features_df], axis=1)\n\nrank_features = np.concatenate(rank_features, axis=0)\nrank_features_df = pd.DataFrame(rank_features)\nrank_features_df.columns = [f\'rank_feature_{i}\' for i in range(rank_features.shape[1])]\ntrain = pd.concat([train, rank_features_df], axis=1)\n\ndel features, features_df, \\\n    rank_features, rank_features_df\ngc.collect()\n\ntrain.to_csv(\'train_data.csv\', index=False)\nprint(f\'train.shape: {train.shape}\')\nprint(f\'Mean target: {train.target.mean()}\')\ntrain.head(20)')


# ## Preparing test dataset

# In[ ]:


get_ipython().run_cell_magic('time', '', 'ids = []\nquestion_tfidfs = []\nanswer_tfidfs = []\ncandidates_str = []\ntargets = []\ntargets_str = []\nfeatures = []\nrank_features = []\n\nwith open(\'/kaggle/input/tensorflow2-question-answering/simplified-nq-test.jsonl\', \'r\') as json_file:\n    for line in tqdm(json_file):\n        json_data = json.loads(line) \n        \n        # TFIDF for document\n        stop_words = text.ENGLISH_STOP_WORDS.union(["book"])\n        tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words=stop_words)\n        tfidf.fit([json_data[\'document_text\']])\n        \n        # TFIDF for question\n        question = json_data[\'question_text\']\n        question_tfidf = tfidf.transform([question]).todense()\n        \n        doc_tokenized = json_data[\'document_text\'].split(\' \')\n        candidates = json_data[\'long_answer_candidates\']\n        candidates = [c for c in candidates if c[\'top_level\'] == True]\n        \n        cos_d_buf = []\n        euc_d_buf = []\n        lev_d_buf = []\n        \n        for c in candidates:\n            ids.append(str(json_data[\'example_id\']))\n            \n            # TFIDF for candidate answer\n            start_token = c[\'start_token\']\n            end_token = c[\'end_token\']\n            answer = \' \'.join(doc_tokenized[start_token:end_token])\n            answer_tfidf = tfidf.transform([answer]).todense()\n            \n            # Extract some features\n            cos_d = spatial.distance.cosine(question_tfidf, answer_tfidf)\n            euc_d = np.linalg.norm(question_tfidf - answer_tfidf)\n            lev_d = Levenshtein.distance(clean(question), clean(answer))\n            lev_r = Levenshtein.ratio(clean(question), clean(answer))\n            jar_s = Levenshtein.jaro(clean(question), clean(answer))\n            jaw_s = Levenshtein.jaro_winkler(clean(question), clean(answer))\n            tfidf_score = np.sum(question_tfidf*answer_tfidf.T)\n            question_tfidf_sum = np.sum(question_tfidf)\n            answer_tfidf_sum = np.sum(answer_tfidf)\n\n            features.append([\n                cos_d, \n                euc_d, \n                lev_d, \n                lev_r, \n                jar_s, \n                jaw_s, \n                tfidf_score, \n                question_tfidf_sum, \n                answer_tfidf_sum\n            ])\n\n            cos_d_buf.append(cos_d)\n            euc_d_buf.append(euc_d)\n            lev_d_buf.append(lev_d)\n            \n            candidates_str.append(str(start_token) + \':\' + str(end_token))\n        \n        rank_cos_d = np.argsort(cos_d_buf)\n        rank_euc_d = np.argsort(euc_d_buf)\n        rank_lev_d = np.argsort(lev_d_buf)\n        rank_cos_d_ismin = (cos_d_buf == np.nanmin(cos_d_buf)).astype(int)\n        rank_euc_d_ismin = (euc_d_buf == np.nanmin(euc_d_buf)).astype(int)\n        rank_lev_d_ismin = (lev_d_buf == np.nanmin(lev_d_buf)).astype(int)\n        rank_features.append(np.array([rank_cos_d, rank_euc_d, rank_lev_d, \\\n                                       rank_cos_d_ismin, rank_euc_d_ismin, rank_lev_d_ismin]).T)\n        \ntest = pd.DataFrame()\ntest[\'example_id\'] = ids\ntest[\'CandidateString\'] = candidates_str\n\nfeatures = np.array(features)\nfeatures_df = pd.DataFrame(features)\nfeatures_df.columns = [f\'feature_{i}\' for i in range(features.shape[1])]\ntest = pd.concat([test, features_df], axis=1)\n\nrank_features = np.concatenate(rank_features, axis=0)\nrank_features_df = pd.DataFrame(rank_features)\nrank_features_df.columns = [f\'rank_feature_{i}\' for i in range(rank_features.shape[1])]\ntest = pd.concat([test, rank_features_df], axis=1)\n\ndel features, features_df, rank_features, rank_features_df\ngc.collect()\n\ntest.to_csv(\'test_data.csv\', index=False)\nprint(f\'test.shape: {test.shape}\')\ntest.head(10)')


# ## Build the model

# In[ ]:


p_buf = []
n_splits = 4

kf = GroupKFold(
    n_splits=n_splits)

err_buf = []   

cols_to_drop = ['example_id', 'target', 'CorrectString', 'CorrectString_short', 'CandidateString']

X = train.drop(cols_to_drop, axis=1, errors='ignore')
y = train['target'].values
g = train['example_id'].values

X_test = test.drop(cols_to_drop, axis=1, errors='ignore')
id_test = test['example_id'].values

print(f'X.shape: {X.shape}, y.shape: {y.shape}')
print(f'X_test.shape: {X_test.shape}')

n_features = X.shape[1]

lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'max_depth': 16,
    'learning_rate': 0.0055, 
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'verbose': -1,
    'num_threads': 5,
}

for fold_i, (train_index, valid_index) in enumerate(kf.split(X, y, g)):
    print('Fold {}/{}'.format(fold_i + 1, n_splits))
    params = lgb_params.copy() 
    
    X_train, y_train = X.iloc[train_index], y[train_index]
    X_valid, y_valid = X.iloc[valid_index], y[valid_index]

    print(f'X_train.shape: {X_train.shape}, X_valid.shape: {X_valid.shape}')
    feature_names = list(X_train.columns)

    lgb_train = lgb.Dataset(
        X_train, 
        y_train, 
        feature_name=feature_names,
        )
    lgb_train.raw_data = None

    lgb_valid = lgb.Dataset(
        X_valid, 
        y_valid,
        feature_name=feature_names,
        )
    lgb_valid.raw_data = None

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_valid],
        early_stopping_rounds=100, 
        verbose_eval=100, 
    )

    # Feature importance
    if fold_i == 0:
        importance = model.feature_importance()
        model_fnames = model.feature_name()
        tuples = sorted(zip(model_fnames, importance), key=lambda x: x[1])[::-1]
        tuples = [x for x in tuples if x[1] > 0]
        print('Important features:')
        for i in range(20):
            if i < len(tuples):
                print(i, tuples[i])
            else:
                break

    # Evaluate model
    p = model.predict(X.loc[valid_index], num_iteration=model.best_iteration)
    valid_df = train.loc[valid_index]
    valid_df['pred'] = p
    pred_df = valid_df.sort_values('pred', ascending=True).groupby('example_id').tail(1)

    pred_df_long = pred_df[['example_id', 'CorrectString', 'CandidateString']]
    pred_df_long.rename({'CandidateString': 'PredictionString'}, axis=1, inplace=True)
    pred_df_long['example_id'] = pred_df_long['example_id'].apply(lambda x: x + '_long')

    pred_df_short = pred_df[['example_id', 'CorrectString_short', 'CandidateString']]
    pred_df_short.rename({'CorrectString_short': 'CorrectString', 'CandidateString': 'PredictionString'},                          axis=1, inplace=True)
    pred_df_short['example_id'] = pred_df_short['example_id'].apply(lambda x: x + '_short')
    pred_df_short['PredictionString'] = ''

    pred_df = pd.concat([pred_df_long, pred_df_short], axis=0).sort_values('example_id')
#     print(pred_df.head(20))

    err = f1_score(pred_df['CorrectString'].values, pred_df['PredictionString'].values, average='micro')
    print('{} F1: {}'.format(fold_i, err))
    
    # Inference on test data
    p_test = model.predict(X_test[feature_names], num_iteration=model.best_iteration)
    p_buf.append(p_test)
    err_buf.append(err)

#     if fold_i >= 0: # Comment this to run several folds
#         break

    del model, lgb_train, lgb_valid, p
    gc.collect()


# In[ ]:


err_mean = np.mean(err_buf)
err_std = np.std(err_buf)
print('F1 = {:.4f} +/- {:.4f}'.format(err_mean, err_std))


# ## Prepare submission

# In[ ]:


valid_df = train.loc[valid_index]
test['pred'] = np.mean(p_buf, axis=0)
pred_df = test.sort_values('pred', ascending=True).groupby('example_id').tail(1)

pred_df_long = pred_df[['example_id', 'CandidateString']]
pred_df_long.rename({'CandidateString': 'PredictionString'}, axis=1, inplace=True)
pred_df_long['example_id'] = pred_df_long['example_id'].apply(lambda x: str(x) + '_long')

pred_df_short = pred_df[['example_id', 'CandidateString']]
pred_df_short.rename({'CandidateString': 'PredictionString'}, axis=1, inplace=True)
pred_df_short['example_id'] = pred_df_short['example_id'].apply(lambda x: str(x) + '_short')
pred_df_short['PredictionString'] = ''

subm = pd.concat([pred_df_long, pred_df_short], axis=0).sort_values('example_id')
subm.to_csv('submission.csv', index=False)
print(f'subm.shape: {subm.shape}')
print(subm.head(20))


# In[ ]:




