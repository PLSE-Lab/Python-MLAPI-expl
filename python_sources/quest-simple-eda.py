#!/usr/bin/env python
# coding: utf-8

# ## Overview
# 
# The purpose of this kernel is to take a look at the data, come up with some insights, and attempt to create a predictive model or two. This notebook is still **very** raw. I will work on it as my very limited time permits, and hope to expend it in the upcoming days and weeks.
# 
# ## Packages
# 
# First, let's load a few useful Python packages. This section will keep growing in subsequent versions of this EDA.

# In[ ]:


import os
import numpy as np
import pandas as pd
import time
from tqdm import tqdm

from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from scipy.sparse import hstack
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from tqdm import tqdm_notebook, tqdm
from scipy import stats

import nltk
from nltk.corpus import stopwords
import string
import gc

from scipy.sparse import hstack

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# This is a kernels-only competition, which means that we can only use tools and data that are available to us in a single Kaggle kernel. The Pyhon libraries that are available to us are the standard Kaggle kernels compute environment. So let's take a look at the data that's available to us: 

# In[ ]:


import os
print(os.listdir("../input/google-quest-challenge"))


# Let's now load the datasets.

# In[ ]:


train = pd.read_csv('../input/google-quest-challenge/train.csv').fillna(' ')
test = pd.read_csv('../input/google-quest-challenge/test.csv').fillna(' ')
sample_submission = pd.read_csv('../input/google-quest-challenge/sample_submission.csv')


# The metric for this competitiomn is Spearman Correlation, and we will define it here for later use:

# In[ ]:


def spearman_corr(y_true, y_pred):
        if np.ndim(y_pred) == 2:
            corr = np.mean([stats.spearmanr(y_true[:, i], y_pred[:, i])[0] for i in range(y_true.shape[1])])
        else:
            corr = stats.spearmanr(y_true, y_pred)[0]
        return corr


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.shape


# In[ ]:


test.shape


# This is a **very** small datset, especially for NLP competitions. Furthermore, we ahve 30 different target variables. It's very likely that many of those vriables will be hard to model with any high degree of accuracy.
# 
# Let's not take a look at what kind of target valiables we have:

# In[ ]:


targets = list(sample_submission.columns[1:])
targets


# Most of these targets are fairly self-explanatory. let's look at tehir distributions in the train dataset.

# In[ ]:


train[targets].describe()


# We see that all targets are have values between 0 and 1. Other than that, their distributions vary - dramatically. However, there is only a handful of discrete valuas that each one of the target variables seems to attain. Let's take a closer look.

# In[ ]:


np.unique(train[targets].values, return_counts=True)


# In[ ]:


np.unique(train[targets].values).shape


# So there are really only 25 discrete values that we have to deal with.

# Let's take a look at some of these 

# In[ ]:


x= np.unique(train['question_asker_intent_understanding'].values, return_counts=True)[0]
y= np.unique(train['question_asker_intent_understanding'].values, return_counts=True)[1]
plt.bar(x, y, align='center', width=0.05)


# In[ ]:


x= np.unique(train['question_body_critical'].values, return_counts=True)[0]
y= np.unique(train['question_body_critical'].values, return_counts=True)[1]
plt.bar(x, y, align='center', width=0.05)


# In[ ]:


x= np.unique(train['question_not_really_a_question'].values, return_counts=True)[0]
y= np.unique(train['question_not_really_a_question'].values, return_counts=True)[1]
plt.bar(x, y, align='center', width=0.05)


# In[ ]:


x= np.unique(train['question_conversational'].values, return_counts=True)[0]
y= np.unique(train['question_conversational'].values, return_counts=True)[1]
plt.bar(x, y, align='center', width=0.05)


# The metric for this competition is Spearman Correlation. It would be interesting to see how correlated various target columns are.

# In[ ]:


corr = train[targets].corr()
corr.style.background_gradient(cmap='coolwarm')


# So that is very interesting. Some of the targets are **extremely** correlated, such as 'question_type_instructions' and 'answer_type_instructions'. In that case this seems quite intutively obvious. 

# For EDA and later modeling, it might be a good idea to create some metafeatures. This work is partly based on SRK's great EDAs, and [this one](http://www.kaggle.com/sudalairajkumar/simple-feature-engg-notebook-spooky-author) in particular. The metafeatures that we'll create are:
# 
# 
# * Number of words in the question_title
# * Number of words in the question_body
# * Number of words in the answer
# 
# * Number of unique words in the question_title
# * Number of unique words in the question_body
# * Number of unique words in the answer
# 
# * Number of characters in the question_title
# * Number of characters in the question_body
# * Number of characters in the answer
# 
# * Number of stopwords in question_title
# * Number of stopwords in question_body
# * Number of stopwords in answer
# 
# * Number of punctuations in question_title
# * Number of punctuations in question_body
# * Number of punctuations in answer
# 
# * Number of upper case words in question_title
# * Number of upper case words in question_body
# * Number of upper case words in answer

# In[ ]:


eng_stopwords = set(stopwords.words("english"))


## Number of words in the text ##
train["question_title_num_words"] = train["question_title"].apply(lambda x: len(str(x).split()))
test["question_title_num_words"] = test["question_title"].apply(lambda x: len(str(x).split()))
train["question_body_num_words"] = train["question_body"].apply(lambda x: len(str(x).split()))
test["question_body_num_words"] = test["question_body"].apply(lambda x: len(str(x).split()))
train["answer_num_words"] = train["answer"].apply(lambda x: len(str(x).split()))
test["answer_num_words"] = test["answer"].apply(lambda x: len(str(x).split()))


## Number of unique words in the text ##
train["question_title_num_unique_words"] = train["question_title"].apply(lambda x: len(set(str(x).split())))
test["question_title_num_unique_words"] = test["question_title"].apply(lambda x: len(set(str(x).split())))
train["question_body_num_unique_words"] = train["question_body"].apply(lambda x: len(set(str(x).split())))
test["question_body_num_unique_words"] = test["question_body"].apply(lambda x: len(set(str(x).split())))
train["answer_num_unique_words"] = train["answer"].apply(lambda x: len(set(str(x).split())))
test["answer_num_unique_words"] = test["answer"].apply(lambda x: len(set(str(x).split())))

## Number of characters in the text ##
train["question_title_num_chars"] = train["question_title"].apply(lambda x: len(str(x)))
test["question_title_num_chars"] = test["question_title"].apply(lambda x: len(str(x)))
train["question_body_num_chars"] = train["question_body"].apply(lambda x: len(str(x)))
test["question_body_num_chars"] = test["question_body"].apply(lambda x: len(str(x)))
train["answer_num_chars"] = train["answer"].apply(lambda x: len(str(x)))
test["answer_num_chars"] = test["answer"].apply(lambda x: len(str(x)))

## Number of stopwords in the text ##
train["question_title_num_stopwords"] = train["question_title"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
test["question_title_num_stopwords"] = test["question_title"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
train["question_body_num_stopwords"] = train["question_body"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
test["question_body_num_stopwords"] = test["question_body"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
train["answer_num_stopwords"] = train["answer"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
test["answer_num_stopwords"] = test["answer"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

## Number of punctuations in the text ##
train["question_title_num_punctuations"] =train['question_title'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
test["question_title_num_punctuations"] =test['question_title'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
train["question_body_num_punctuations"] =train['question_body'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
test["question_body_num_punctuations"] =test['question_body'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
train["answer_num_punctuations"] =train['answer'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
test["answer_num_punctuations"] =test['answer'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

## Number of title case words in the text ##
train["question_title_num_words_upper"] = train["question_title"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
test["question_title_num_words_upper"] = test["question_title"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
train["question_body_num_words_upper"] = train["question_body"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
test["question_body_num_words_upper"] = test["question_body"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
train["answer_num_words_upper"] = train["answer"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
test["answer_num_words_upper"] = test["answer"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))


# In[ ]:


features = ['question_title_num_words', 'question_body_num_words', 'answer_num_words', 'question_title_num_unique_words', 'question_body_num_unique_words', 'answer_num_unique_words',
           'question_title_num_chars', 'question_body_num_chars', 'answer_num_chars', 'question_title_num_stopwords', 'question_body_num_stopwords', 'question_title_num_punctuations',
           'question_body_num_punctuations', 'answer_num_punctuations', 'question_title_num_words_upper', 'question_body_num_words_upper', 'answer_num_words_upper']


# In[ ]:


plt.figure(figsize=(12,8))
sns.violinplot(data=train['question_body_num_words'])
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
sns.violinplot(data=train['question_body_num_chars'])
plt.show()


# In[ ]:



plt.figure(figsize=(12,8))
sns.violinplot(data=train['answer_num_chars'])
plt.show()


# In[ ]:


X_train = train[features].values
X_test = test[features].values
class_names_2 = [class_name+'_2' for class_name in targets]
for class_name in targets:
    train[class_name+'_2'] = (train[class_name].values >= 0.5)*1


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nsubmission_1 = pd.DataFrame.from_dict({\'qa_id\': test[\'qa_id\']})\n\nscores = []\nspearman_scores = []\n\nfor class_name in tqdm_notebook(targets):\n    print(class_name)\n    Y = train[class_name]\n    \n    n_splits = 3\n    kf = KFold(n_splits=n_splits, random_state=47)\n\n    train_oof = np.zeros((X_train.shape[0], ))\n    test_preds = 0\n    \n    score = 0\n\n    for jj, (train_index, val_index) in enumerate(kf.split(X_train)):\n        #print("Fitting fold", jj+1)\n        train_features = X_train[train_index]\n        train_target = Y[train_index]\n\n        val_features = X_train[val_index]\n        val_target = Y[val_index]\n\n        model = Ridge()\n        model.fit(train_features, train_target)\n        val_pred = model.predict(val_features)\n        train_oof[val_index] = val_pred\n        #print("Fold auc:", roc_auc_score(val_target, val_pred))\n        #score += roc_auc_score(val_target, val_pred)/n_splits\n\n        test_preds += model.predict(X_test)/n_splits\n        del train_features, train_target, val_features, val_target\n        gc.collect()\n        \n    model = Ridge()\n    model.fit(X_train, Y)\n    \n    preds = model.predict(X_test)\n    mms = MinMaxScaler(copy=True, feature_range=(0, 1))\n    preds = mms.fit_transform(preds.reshape(-1, 1)).flatten()\n    submission_1[class_name] = (preds+0.00005)/1.0001\n        \n    score = roc_auc_score(train[class_name+\'_2\'], train_oof) \n    \n    \n    spearman_score = spearman_corr(train[class_name], train_oof)\n    print("spearman_corr:", spearman_score)\n    print("auc:", score, "\\n")\n    spearman_scores.append(spearman_score)\n    \n    scores.append(score)\n    \nprint("Mean auc:", np.mean(scores))\nprint("Mean spearman_scores", np.mean(spearman_scores))')


# In[ ]:


HistGradientBoostingRegressor()


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nsubmission_2 = pd.DataFrame.from_dict({\'qa_id\': test[\'qa_id\']})\n\nscores = []\nspearman_scores = []\n\nfor class_name in tqdm_notebook(targets):\n    print(class_name)\n    Y = train[class_name]\n    \n    n_splits = 3\n    kf = KFold(n_splits=n_splits, random_state=47)\n\n    train_oof = np.zeros((X_train.shape[0], ))\n    test_preds = 0\n    \n    score = 0\n\n    for jj, (train_index, val_index) in enumerate(kf.split(X_train)):\n        #print("Fitting fold", jj+1)\n        train_features = X_train[train_index]\n        train_target = Y[train_index]\n\n        val_features = X_train[val_index]\n        val_target = Y[val_index]\n\n        model = HistGradientBoostingRegressor(max_depth=5)\n        model.fit(train_features, train_target)\n        val_pred = model.predict(val_features)\n        train_oof[val_index] = val_pred\n        #print("Fold auc:", roc_auc_score(val_target, val_pred))\n        #score += roc_auc_score(val_target, val_pred)/n_splits\n\n        test_preds += model.predict(X_test)/n_splits\n        del train_features, train_target, val_features, val_target\n        gc.collect()\n        \n    model = HistGradientBoostingRegressor(max_depth=5)\n    model.fit(X_train, Y)\n    \n    preds = model.predict(X_test)\n    mms = MinMaxScaler(copy=True, feature_range=(0, 1))\n    preds = mms.fit_transform(preds.reshape(-1, 1)).flatten()\n    submission_2[class_name] = (preds+0.00005)/1.0001\n        \n    score = roc_auc_score(train[class_name+\'_2\'], train_oof) \n    \n    \n    spearman_score = spearman_corr(train[class_name], train_oof)\n    print("spearman_corr:", spearman_score)\n    print("auc:", score, "\\n")\n    spearman_scores.append(spearman_score)\n    \n    scores.append(score)\n    \nprint("Mean auc:", np.mean(scores))\nprint("Mean spearman_scores", np.mean(spearman_scores))')


# In[ ]:


submission_1.head()


# In[ ]:


submission_2.head()


# In[ ]:


submission = submission_1.copy()
submission[targets] = 0.1*submission_1[targets].values + 0.9*submission_2[targets].values
submission.head()


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




