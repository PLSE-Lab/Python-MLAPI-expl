#!/usr/bin/env python
# coding: utf-8

# This is a Ridge Linear Regression QUEST version of the following Jigsaw Toxic Comments kernel: https://www.kaggle.com/tunguz/logistic-regression-with-words-and-char-n-grams
# 
# Which in turn was forked from the followiong kernel: https://www.kaggle.com/thousandvoices/logistic-regression-with-words-and-char-n-grams

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from scipy.sparse import hstack
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm_notebook, tqdm
from scipy import stats

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import gc
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


def spearman_corr(y_true, y_pred):
        if np.ndim(y_pred) == 2:
            corr = np.mean([stats.spearmanr(y_true[:, i], y_pred[:, i])[0] for i in range(y_true.shape[1])])
        else:
            corr = stats.spearmanr(y_true, y_pred)[0]
        return corr


# In[ ]:


train = pd.read_csv('../input/google-quest-challenge/train.csv').fillna(' ')
test = pd.read_csv('../input/google-quest-challenge/test.csv').fillna(' ')
train.head()


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


np.unique(train['category'].values)


# In[ ]:


train_text_1 = train['question_body']
test_text_1 = test['question_body']
all_text_1 = pd.concat([train_text_1, test_text_1])

train_text_2 = train['answer']
test_text_2 = test['answer']
all_text_2 = pd.concat([train_text_2, test_text_2])

train_text_3 = train['question_title']
test_text_3 = test['question_title']
all_text_3 = pd.concat([train_text_3, test_text_3])


# In[ ]:


sample_submission = pd.read_csv('../input/google-quest-challenge/sample_submission.csv').fillna(' ')
sample_submission.head()


# In[ ]:


class_names = list(sample_submission.columns[1:])
class_names


# In[ ]:


class_names_q = class_names[:21]
class_names_a = class_names[21:]
class_names_a


# In[ ]:


class_names_2 = [class_name+'_2' for class_name in class_names]
for class_name in class_names:
    train[class_name+'_2'] = (train[class_name].values >= 0.5)*1


# In[ ]:


get_ipython().run_cell_magic('time', '', "word_vectorizer = TfidfVectorizer(\n    sublinear_tf=True,\n    strip_accents='unicode',\n    analyzer='word',\n    token_pattern=r'\\w{1,}',\n    stop_words='english',\n    ngram_range=(1, 2),\n    max_features=80000)\nword_vectorizer.fit(all_text_1)\ntrain_word_features_1 = word_vectorizer.transform(train_text_1)\ntest_word_features_1 = word_vectorizer.transform(test_text_1)\n\nword_vectorizer = TfidfVectorizer(\n    sublinear_tf=True,\n    strip_accents='unicode',\n    analyzer='word',\n    token_pattern=r'\\w{1,}',\n    stop_words='english',\n    ngram_range=(1, 2),\n    max_features=80000)\nword_vectorizer.fit(all_text_2)\ntrain_word_features_2 = word_vectorizer.transform(train_text_2)\ntest_word_features_2 = word_vectorizer.transform(test_text_2)\n\nword_vectorizer = TfidfVectorizer(\n    sublinear_tf=True,\n    strip_accents='unicode',\n    analyzer='word',\n    token_pattern=r'\\w{1,}',\n    stop_words='english',\n    ngram_range=(1, 2),\n    max_features=80000)\nword_vectorizer.fit(all_text_3)\ntrain_word_features_3 = word_vectorizer.transform(train_text_3)\ntest_word_features_3 = word_vectorizer.transform(test_text_3)\n\nchar_vectorizer = TfidfVectorizer(\n    sublinear_tf=True,\n    strip_accents='unicode',\n    analyzer='char',\n    stop_words='english',\n    ngram_range=(1, 4),\n    max_features=47000)\nchar_vectorizer.fit(all_text_1)\ntrain_char_features_1 = char_vectorizer.transform(train_text_1)\ntest_char_features_1 = char_vectorizer.transform(test_text_1)\n\nchar_vectorizer = TfidfVectorizer(\n    sublinear_tf=True,\n    strip_accents='unicode',\n    analyzer='char',\n    stop_words='english',\n    ngram_range=(1, 4),\n    max_features=47000)\nchar_vectorizer.fit(all_text_2)\ntrain_char_features_2 = char_vectorizer.transform(train_text_2)\ntest_char_features_2 = char_vectorizer.transform(test_text_2)\n\nchar_vectorizer = TfidfVectorizer(\n    sublinear_tf=True,\n    strip_accents='unicode',\n    analyzer='char',\n    stop_words='english',\n    ngram_range=(1, 4),\n    max_features=47000)\nchar_vectorizer.fit(all_text_3)\ntrain_char_features_3 = char_vectorizer.transform(train_text_3)\ntest_char_features_3 = char_vectorizer.transform(test_text_3)\n\ntrain_features_1 = hstack([train_char_features_1, train_word_features_1, train_char_features_3, train_word_features_3])\ntest_features_1 = hstack([test_char_features_1, test_word_features_1, test_char_features_3, test_word_features_3])\ntrain_features_2 = hstack([train_char_features_2, train_word_features_2])\ntest_features_2 = hstack([test_char_features_2, test_word_features_2])")


# In[ ]:


train_features_1= train_features_1.tocsr()
train_features_2= train_features_2.tocsr()


# In[ ]:


alphas = {'question_asker_intent_understanding': 40,
         'question_body_critical':7,
          'question_conversational':35,
          'question_expect_short_answer':65,
          'question_fact_seeking':10,
          'question_has_commonly_accepted_answer':25,
          'question_interestingness_others':50,
          'question_interestingness_self':30,
          'question_multi_intent':7,
          'question_not_really_a_question':55,
          'question_opinion_seeking':15,
          'question_type_choice':4,
          'question_type_compare':30,
          'question_type_consequence':45,
          'question_type_definition':60,
          'question_type_entity':11,
          'question_type_instructions':6,
          'question_type_procedure':40,
          'question_type_reason_explanation':13,
          'question_type_spelling':1,
          'question_well_written':8,
          'answer_helpful':30,
          'answer_level_of_information':8,
          'answer_plausible':20,
          'answer_relevance':60,
          'answer_satisfaction':11,
          'answer_type_instructions':3,
          'answer_type_procedure':25,
          'answer_type_reason_explanation':3,
          'answer_well_written':25
         }


# In[ ]:


get_ipython().run_cell_magic('time', '', 'submission = pd.DataFrame.from_dict({\'qa_id\': test[\'qa_id\']})\n\ntrain_preds = []\ntest_preds = []\nscores = []\nspearman_scores = []\n\nfor class_name in tqdm_notebook(class_names_q):\n    print(class_name)\n    Y = train[class_name]\n    \n    n_splits = 3\n    kf = KFold(n_splits=n_splits, random_state=47)\n\n    train_oof_1 = np.zeros((train_features_1.shape[0], ))\n    test_preds_1 = 0\n    \n    score = 0\n\n    for jj, (train_index, val_index) in enumerate(kf.split(train_features_1)):\n        #print("Fitting fold", jj+1)\n        train_features = train_features_1[train_index]\n        train_target = Y[train_index]\n\n        val_features = train_features_1[val_index]\n        val_target = Y[val_index]\n\n        model = Ridge(alpha = alphas[class_name])\n        model.fit(train_features, train_target)\n        val_pred = model.predict(val_features)\n        train_oof_1[val_index] = val_pred\n        #print("Fold auc:", roc_auc_score(val_target, val_pred))\n        #spearman_corr\n        #score += roc_auc_score(val_target, val_pred)/n_splits\n\n        test_preds_1 += model.predict(test_features_1)/n_splits\n        del train_features, train_target, val_features, val_target\n        gc.collect()\n        \n    model = Ridge(alpha = alphas[class_name])\n    model.fit(train_features_1, Y)\n    #print(test_preds_1.max())\n    #print(test_preds_1.min())\n    mms = MinMaxScaler(copy=True, feature_range=(0, 1))\n    test_preds_1 = mms.fit_transform(test_preds_1.reshape(-1, 1)).flatten()\n    preds = model.predict(test_features_1)\n    mms = MinMaxScaler(copy=True, feature_range=(0, 1))\n    preds = mms.fit_transform(preds.reshape(-1, 1)).flatten()\n    submission[class_name] = (0.75*test_preds_1+0.25*preds+0.000005)/1.00001\n    spearman_score = spearman_corr(train[class_name], train_oof_1)\n    print("spearman_corr:", spearman_score) \n    spearman_scores.append(spearman_score)\n    score = roc_auc_score(train[class_name+\'_2\'], train_oof_1)    \n    print("auc:", score, "\\n")\n    train_preds.append(train_oof_1)\n    test_preds.append(test_preds_1)\n    scores.append(score)\n    \n    ')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nfor class_name in tqdm_notebook(class_names_a):\n    print(class_name)\n    Y = train[class_name]\n    \n    n_splits = 3\n    kf = KFold(n_splits=n_splits, random_state=47)\n\n    train_oof_2 = np.zeros((train_features_2.shape[0], ))\n    test_preds_2 = 0\n    \n    score = 0\n\n    for jj, (train_index, val_index) in enumerate(kf.split(train_features_1)):\n        #print("Fitting fold", jj+1)\n        train_features = train_features_2[train_index]\n        train_target = Y[train_index]\n\n        val_features = train_features_2[val_index]\n        val_target = Y[val_index]\n\n        model = Ridge(alpha = alphas[class_name])\n        model.fit(train_features, train_target)\n        val_pred = model.predict(val_features)\n        train_oof_2[val_index] = val_pred\n        #print("Fold auc:", roc_auc_score(val_target, val_pred))\n        #score += roc_auc_score(val_target, val_pred)/n_splits\n\n        test_preds_2 += model.predict(test_features_2)/n_splits\n        del train_features, train_target, val_features, val_target\n        gc.collect()\n        \n    model = Ridge(alpha = alphas[class_name])\n    model.fit(train_features_2, Y)\n    print(test_preds_2.max())\n    \n    preds = model.predict(test_features_2)\n    mms = MinMaxScaler(copy=True, feature_range=(0, 1))\n    test_preds_2 = mms.fit_transform(test_preds_2.reshape(-1, 1)).flatten()\n    mms = MinMaxScaler(copy=True, feature_range=(0, 1))\n    preds = mms.fit_transform(preds.reshape(-1, 1)).flatten()\n    submission[class_name] = (0.75*test_preds_2+0.25*preds+0.000005)/1.00001\n        \n    score = roc_auc_score(train[class_name+\'_2\'], train_oof_2) \n    \n    \n    spearman_score = spearman_corr(train[class_name], train_oof_2)\n    print("spearman_corr:", spearman_score)\n    print("auc:", score, "\\n")\n    spearman_scores.append(spearman_score)\n    \n    train_preds.append(train_oof_2)\n    test_preds.append(test_preds_2)\n    scores.append(score)')


# In[ ]:


print("Mean auc:", np.mean(scores))
print("Mean spearman_scores", np.mean(spearman_scores))


# In[ ]:


submission.to_csv('submission.csv', index=False)
submission.head()


# In[ ]:


submission[class_names].values.max()


# In[ ]:


submission[class_names].values.min()

