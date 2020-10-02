#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from vowpalwabbit import pyvw

from gensim.parsing import preprocessing as prep
from collections import defaultdict
from sklearn.model_selection import train_test_split


# In[ ]:


train = pd.read_csv('../input/train.csv')
train.head()


# In[ ]:


plt.hist(train['target']);


# In[ ]:


class Tokenizer(object):
    def __call__(self, doc): 
        striped = prep.strip_punctuation(doc)
        striped = prep.strip_tags(striped)
        striped = prep.strip_multiple_whitespaces(striped).lower()
        return striped


# In[ ]:


class FilterRareWords(object):
    def __init__(self):
        self.cv = defaultdict(int)
    def fit(self, texts):
        for text in texts:
            for word in text.split():
                self.cv[word] += 1
    def __call__(self, text):
        return ' '.join([self.filter_word(word) for word in text.split()])
    def filter_word(self, word):
        return '' if self.cv[word] < 2 else word


# In[ ]:


tokenizer = Tokenizer()
filter_words = FilterRareWords()


# In[ ]:


train[train['target'] == 1].head()


# In[ ]:


train['question_text'] = train['question_text'].apply(tokenizer)

filter_words.fit(train['question_text'])
train['question_text'] = train['question_text'].apply(filter_words)


# In[ ]:


pos_weight = train['target'].sum() / train.shape[0]


# In[ ]:


def make_vw_feature_line(label, importance, text):
    return '{} {} |text {}'.format(label, importance, text)

def make_vw_corpus(texts, labels):
    for text, label in zip(texts, labels):
        if label == 1.0:
            cur_feautre = make_vw_feature_line('1', 1 - pos_weight, text)
        else:
            cur_feautre = make_vw_feature_line('-1', pos_weight, text)
        yield cur_feautre


# In[ ]:


X_train, X_test = train_test_split(train, test_size=0.1, shuffle=True, random_state=42)


# In[ ]:


vw = pyvw.vw(
    quiet=True,
    loss_function='logistic',
    link='logistic',
    b=29,
    ngram=2,
    skips=1,
    random_seed=42,
    l1=3.4742122764e-09,
    l2=1.24232077629e-11,
    learning_rate=0.751849318433,
)


# In[ ]:


def get_pred(feature):
    ex = vw.example(feature)
    pred = vw.predict(ex)
    ex.finish()
    return pred


# In[ ]:


get_ipython().run_cell_magic('time', '', "for fit_iter in range(3):\n    for num, feature in enumerate(make_vw_corpus(X_train['question_text'], X_train['target'])):\n        ex = vw.example(feature)\n        vw.learn(ex)\n        ex.finish()\n        \n    print('pass num {} done'.format(fit_iter))")


# In[ ]:


pred = np.array([get_pred(x) for x in make_vw_corpus(X_test['question_text'], X_test['target'])])


# In[ ]:


thresholds = np.linspace(0, 1, 100)
f1_scores = [metrics.f1_score(X_test['target'], pred > threshold) for threshold in thresholds]


# In[ ]:


plt.plot(thresholds, f1_scores)
plt.grid(True)
plt.show()


# In[ ]:


print('best f1 score is {} with threshold {}'.format(np.max(f1_scores), thresholds[np.argmax(f1_scores)]))


# In[ ]:


test = pd.read_csv('../input/test.csv')

test['question_text'] = test['question_text'].apply(tokenizer)
test['question_text'] = test['question_text'].apply(filter_words)

pred = np.array([get_pred(x) for x in make_vw_corpus(test['question_text'], [1] * len(test))])

example = pd.read_csv('../input/sample_submission.csv')
example['prediction'] = (pred > thresholds[np.argmax(f1_scores)]).astype(int)
example.to_csv('submission.csv', index=False)


# In[ ]:


plt.hist(example['prediction']);


# In[ ]:




