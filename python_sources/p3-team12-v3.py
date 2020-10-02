#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


train_df = pd.read_csv("../input/train.csv")
train_df = train_df[['id','comment_text', 'target']]
test_df = pd.read_csv("../input/test.csv")


# In[3]:


from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence
  
train_df['comment_text'] = train_df['comment_text'].apply(stemming)


# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer
import re, string
re_tok = re.compile(f'([{string.punctuation}])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

vect = TfidfVectorizer(input="content", 
                encoding="utf-8", 
                decode_error="strict", 
                strip_accents="unicode", 
                lowercase=True, 
                preprocessor=None, 
                tokenizer=tokenize, 
                analyzer="word", 
                stop_words=None, 
                token_pattern="(?u)\b\w\w+\b", 
                ngram_range=(1, 2), 
                max_df=0.9, 
                min_df=3, 
                max_features=None, 
                vocabulary=None, 
                binary=False, 
                norm="l2", 
                use_idf=1, 
                smooth_idf=1, 
                sublinear_tf=1)


# In[5]:


X = vect.fit_transform(train_df["comment_text"])
y = np.where(train_df['target'] >= 0.5, 1, 0)

test_X = vect.transform(test_df["comment_text"])


# In[6]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier  

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, random_state=42)


# In[8]:


get_ipython().run_cell_magic('time', '', 'from sklearn.linear_model import LogisticRegression\n\nlr = LogisticRegression(penalty="l2", \n                                             dual=False, \n                                             tol=0.0001, \n                                             C=1.0, \n                                             fit_intercept=True, \n                                             intercept_scaling=1, \n                                             class_weight="balanced", \n                                             random_state=None, \n                                             solver="liblinear", \n                                             max_iter=100, \n                                             multi_class="auto", \n                                             verbose=0, \n                                             warm_start=False, \n                                             n_jobs=None)\n\nlr.fit(X_train, y_train)\ny_pred = lr.predict(X_test)')


# In[9]:


cv_accuracy = cross_val_score(
    LogisticRegression(C=5, random_state=42, solver='sag', max_iter=1000, n_jobs=-1), 
    X, y, cv=5, scoring='roc_auc'
)
print(cv_accuracy)
print(cv_accuracy.mean())


# In[10]:


accuracy_score(y_test, y_pred)


# In[11]:


prediction = lr.predict_proba(test_X)[:,1]


# In[12]:


submission = pd.read_csv("../input/sample_submission.csv")
submission['prediction'] = prediction
submission.to_csv('submission.csv', index=False)

