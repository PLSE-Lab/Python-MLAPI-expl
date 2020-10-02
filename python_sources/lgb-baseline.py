#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


import nltk
from nltk.corpus import stopwords
import string

eng_stopwords = set(stopwords.words("english"))

## Number of words in the text ##
train["num_words"] = train["question_text"].apply(lambda x: len(str(x).split()))
test["num_words"] = test["question_text"].apply(lambda x: len(str(x).split()))

## Number of unique words in the text ##
train["num_unique_words"] = train["question_text"].apply(lambda x: len(set(str(x).split())))
test["num_unique_words"] = test["question_text"].apply(lambda x: len(set(str(x).split())))

## Number of characters in the text ##
train["num_chars"] = train["question_text"].apply(lambda x: len(str(x)))
test["num_chars"] = test["question_text"].apply(lambda x: len(str(x)))

## Number of stopwords in the text ##
train["num_stopwords"] = train["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
test["num_stopwords"] = test["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

## Number of punctuations in the text ##
train["num_punctuations"] =train['question_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
test["num_punctuations"] =test['question_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

## Number of title case words in the text ##
train["num_words_upper"] = train["question_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
test["num_words_upper"] = test["question_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

## Number of title case words in the text ##
train["num_words_title"] = train["question_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
test["num_words_title"] = test["question_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

## Average length of the words in the text ##
train["mean_word_len"] = train["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test["mean_word_len"] = test["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))


# In[ ]:


train_text = train['question_text']
test_text = test['question_text']
all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=5000)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)


# In[ ]:


eng_features = ['num_words', 'num_unique_words', 'num_chars', 
                'num_stopwords', 'num_punctuations', 'num_words_upper', 
                'num_words_title', 'mean_word_len']
train_ = train[eng_features]
train_.head()


# In[ ]:


from scipy.sparse import hstack, csr_matrix
train_ = hstack((csr_matrix(train_), train_word_features))
print(train_.shape)


# In[ ]:


test_ = test[eng_features]
test_ = hstack((csr_matrix(test_), test_word_features))
print(test_.shape)


# In[ ]:


from sklearn.model_selection import train_test_split
y = train['target']
X_tr, X_va, y_tr, y_va = train_test_split(train_, y, test_size=0.2, random_state=42)
print(X_tr.shape, X_va.shape)


# In[ ]:


y_va.value_counts()


# In[ ]:


get_ipython().run_cell_magic('time', '', "import lightgbm as lgb\n\nfrom sklearn.metrics import f1_score\n\ndef lgb_f1_score(y_hat, data):\n    y_true = data.get_label()\n    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities\n    return 'f1', f1_score(y_true, y_hat), True\n\nparams = {'application': 'binary',\n          'metric': 'binary_logloss',\n          'learning_rate': 0.05,   \n          'max_depth': 9,\n          'num_leaves': 100,\n          'verbosity': -1,\n          'data_random_seed': 3,\n          'bagging_fraction': 0.8,\n          'feature_fraction': 0.4,\n          'nthread': 16,\n          'lambda_l1': 1,\n          'lambda_l2': 1,\n          'num_rounds': 2700,\n          'verbose_eval': 100}\n\nd_train = lgb.Dataset(X_tr, label=y_tr.values)\nd_valid = lgb.Dataset(X_va, label=y_va.values)\nprint('Train LGB')\nnum_rounds = params.pop('num_rounds')\nverbose_eval = params.pop('verbose_eval')\nmodel = lgb.train(params,\n                  train_set=d_train,\n                  num_boost_round=num_rounds,\n                  valid_sets=[d_train, d_valid],\n                  verbose_eval=verbose_eval,\n                  valid_names=['train', 'val'],\n                  feval=lgb_f1_score)\nprint('Predict')\npred_test_va = model.predict(X_va)")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'best_threshold = 0.01\nbest_score = 0.0\nfor threshold in range(1, 100):\n    threshold = threshold / 100\n    score = f1_score(y_va, pred_test_va > threshold)\n    if score > best_score:\n        best_threshold = threshold\n        best_score = score\nprint(0.5, f1_score(y_va, pred_test_va > 0.5))\nprint(best_threshold, best_score)\n# 0.24 0.5918758665447358')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'pred_test_y = model.predict(test_)')


# In[ ]:


submit_df = pd.DataFrame({"qid": test["qid"], "prediction": (pred_test_y > best_threshold).astype(np.int)})
submit_df.head()


# In[ ]:


submit_df['prediction'].value_counts()


# In[ ]:


submit_df.to_csv("submission.csv", index=False)

