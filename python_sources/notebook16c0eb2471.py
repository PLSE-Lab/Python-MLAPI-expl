#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import re
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

import sklearn.ensemble
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

# Any results you write to the current directory are saved as output.


# In[ ]:


ENGLISH_STOP_WORDS = set(stopwords.words("english"))
targets = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']


# In[ ]:


space_join = " ".join

def custom_preprocessor(comment):
    comment = re.sub(r'(\w+!)', r'EMOTION', comment)
    comment = re.sub(r'(\n\n*)', '', comment)
    comment = re.sub(r'\d+[.]\d+[.]\d+[.]\d+', '', comment)
    comment = re.sub(r'([.?!;])(\s+)(\w+)', r'\1\3', comment)
    return comment

def custom_tokenizer(comment):
    if comment == "":
        return []
    token_pattern = re.compile(r"(?u)\b\w\w+\b")
    tokens = token_pattern.findall(comment)
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    return list(set(tokens))

def get_nonalphanumeric_tokens(comment):
    if comment == "":
        return []
    return list(set(re.findall(r'[^\w\s\.]', comment)))

def get_datetime(comment):
    searches = re.findall(r'[012][0-9][:][0-9][0-9]\W?\s+[0-9][0-9]\s[\w]+', comment)
    if len(searches) > 0:
        return 1
    return 0

def get_random_indx(df, type_of_question):
    return np.random.choice(df.loc[df[type_of_question] == 1].index)

def get_sum(row):
    sum_target = 0.0
    for t in targets:
        sum_target += row[t]
    return sum_target


# In[ ]:


train = pd.read_csv('../input/train.csv')
train['unlabeled'] = train.apply(lambda x: get_sum(x), axis=1)


# In[ ]:


train['preprocessed_text'] = train['comment_text'].apply(lambda text: custom_preprocessor(text))
train['tokens'] = train['preprocessed_text'].apply(lambda text: custom_tokenizer(text))

train['n_vocab']  = train['preprocessed_text'].apply(lambda row: len(sorted(set(row))))
train['n_tokens'] = train['tokens'].apply(lambda row: len(row))

train['non_alphas'] = train['preprocessed_text'].apply(lambda text: get_nonalphanumeric_tokens(text))
train['n_non_alphas'] = train['non_alphas'].map(len)


# In[ ]:


count_vect = CountVectorizer(min_df=1, max_df=3, max_features = 256)
X_train_counts = count_vect.fit_transform(pd.Series(train['preprocessed_text'].tolist()))

vocabulary = count_vect.get_feature_names()
bow = pd.DataFrame(X_train_counts.toarray(), columns=vocabulary)
bow['id'] = train['id']
train_bow = train.merge(bow, how='inner', on='id')


# In[ ]:


estimators = {}
features = vocabulary + ['n_non_alphas']

for target in targets:
    X = train_bow[features]
    y = train_bow[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    clf = RandomForestClassifier(random_state=0)
    clf = clf.fit(X_train, y_train)
    estimators[target] = clf
    print("{0:<20} {1}".format(target, log_loss(y_test, clf.predict_proba(X_test))))


# In[ ]:


test = pd.read_csv('../input/test.csv')
test.fillna("", inplace=True)

test['preprocessed_text'] = test['comment_text'].apply(lambda text: custom_preprocessor(text))
test['tokens'] = test['preprocessed_text'].apply(lambda text: custom_tokenizer(text))

test['n_vocab']  = test['preprocessed_text'].apply(lambda row: len(sorted(set(row))))
test['n_tokens'] = test['tokens'].apply(lambda row: len(row))

test['non_alphas'] = test['preprocessed_text'].apply(lambda text: get_nonalphanumeric_tokens(text))
test['n_non_alphas'] = test['non_alphas'].map(len)


# In[ ]:


X_test_counts = count_vect.transform(pd.Series(test['preprocessed_text'].tolist()))

X_test = pd.DataFrame(X_test_counts.toarray(), columns=vocabulary)
X_test['n_non_alphas'] = test['n_non_alphas']


# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv')
for key, clf in estimators.items():
    pred = clf.predict_proba(X_test[features])[:, 1]
    submission[key] = pd.Series(pred)


# In[ ]:


pd.isnull(submission).sum()


# In[ ]:


submission.to_csv('submission.csv', index=False)

