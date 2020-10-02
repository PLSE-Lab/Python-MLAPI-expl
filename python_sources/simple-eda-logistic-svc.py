#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
pd.set_option('display.max_colwidth', -1)
from time import time
import re
import string
import os
from pprint import pprint
import collections

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
sns.set(font_scale=1.3)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import warnings
warnings.filterwarnings('ignore')

np.random.seed(37)


# In[ ]:


from nltk.tokenize import TweetTokenizer
import datetime
import lightgbm as lgb
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier


# In[ ]:


from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")


# In[ ]:


import spacy


# In[ ]:


PATH = '../input/'


# In[ ]:


df_train = pd.read_csv(PATH + "train.tsv", sep = '\t')
df_test = pd.read_csv(PATH + "test.tsv", sep = '\t')


# In[ ]:


df_train.head(10)


# There are 5 different sentiments in the training set.
# * 0 - negative
# * 1 - somewhat negative
# * 2 - neutral
# * 3 - somewhat positive
# * 4 - positive

# Some basic EDA. Not much

# In[ ]:


sns.factorplot(x = "Sentiment", data = df_train, kind = 'count', size = 6)


# Around 8529 reviews are split into nearly 156k phrases and each phrase is tagged from 0 to 4. And so, we see far more neutral phrases than polarizing ones. It is likely that the phrases in groups 0 and 4 likely had the most effect on the movie reviews

# In[ ]:


df_train.Sentiment.value_counts()


# In[ ]:


print ("Number of sentences is {0:.0f}.".format(df_train.SentenceId.count()))

print ("Number of unique sentences is {0:.0f}.".format(df_train.SentenceId.nunique()))

print ("Number of phrases is {0:.0f}.".format(df_train.PhraseId.count()))


# In[ ]:


print ("The average length of phrases in the training set is {0:.0f}.".format(np.mean(df_train['Phrase'].apply(lambda x: len(x.split(" "))))))

print ("The average length of phrases in the test set is {0:.0f}.".format(np.mean(df_test['Phrase'].apply(lambda x: len(x.split(" "))))))


# Common trigrams

# In[ ]:


text = ' '.join(df_train.loc[df_train.Sentiment == 0, 'Phrase'].values)


# In[ ]:


Counter([i for i in ngrams(text.split(), 3)]).most_common(5)


# In[ ]:


print (df_train.info())


# In[ ]:


df_train.Phrase.str.len().sort_values(ascending = False)


# In[ ]:


df_train.loc[105155, 'Phrase']


# In[ ]:


df_train.Sentiment.dtype


# In[ ]:


nlp = spacy.load('en',disable=['parser', 'tagger', 'ner'])
def tokenizer(s): 
    return [w.text.lower() for w in nlp(s)]


# In[ ]:


## stemming sentences
sentences = list(df_train.Phrase.values) + list(df_test.Phrase.values)
sentences2 = [[stemmer.stem(word) for word in sentence.split(" ")] for sentence in sentences]
for i in range(len(sentences2)):sentences2[i] = ' '.join(sentences2[i])


# TFIDF to vectorizer the tokens. uni and bigrams used.

# In[ ]:


tfidf = TfidfVectorizer(strip_accents = 'unicode', tokenizer = tokenizer, encoding='utf-8', ngram_range = (1,2), max_df = 0.75, min_df = 3, sublinear_tf = True)


# In[ ]:


_ = tfidf.fit(sentences2)


# In[ ]:


train_phrases2 = [[stemmer.stem(word) for word in sentence.split(" ")] for sentence in list(df_train.Phrase.values)]
for i in range(len(train_phrases2)):train_phrases2[i] = ' '.join(train_phrases2[i])
train_df_flags = tfidf.transform(train_phrases2)


# In[ ]:


test_phrases2 = [[stemmer.stem(word) for word in sentence.split(" ")] for sentence in list(df_test.Phrase.values)]
for i in range(len(test_phrases2)):test_phrases2[i] = ' '.join(test_phrases2[i])
test_df_flags = tfidf.transform(test_phrases2)


# Unlike traditional test_train_split using random initialization, here I am splitting the dataset into train and validation datasets in order.
# Every sentence is broken down into multiple phrases, and so a random split would ensure that starkly similar phrases from the training set would land in the validation set, thereby the validation set performance misleading us into believing that the model generalized well, while all it did was encounter a validation dataset that was mostly a subset of the training dataset.

# In[ ]:


X_train_tf = train_df_flags[0:125000]
X_valid_tf = train_df_flags[125000:]
y_train_tf = (df_train["Sentiment"])[0:125000]
y_valid_tf = (df_train["Sentiment"])[125000:]

print("X_train shape: ", X_train_tf.shape)
print("X_valid shape: ",X_valid_tf.shape)
print("Y_train shape: ",len(y_train_tf))
print("Y_valid shape: ",len(y_valid_tf))


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


scores = cross_val_score(LogisticRegression(C=4, dual=True), X_train_tf, y_train_tf, cv=5)


# In[ ]:


scores


# In[ ]:


np.mean(scores), np.std(scores)


# Using Logistic Regression, through One Vs Rest classifier. This is just a way to model five separate functions/algorithms/models that predicts each class vs the others

# In[ ]:


logistic = LogisticRegression(C=4, dual=True)
ovrm = OneVsRestClassifier(logistic)
ovrm.fit(X_train_tf, y_train_tf)


# In[ ]:


scores = cross_val_score(ovrm, X_train_tf, y_train_tf, scoring='accuracy', n_jobs=-1, cv=3)


# In[ ]:


print (np.mean(scores))
print (np.std(scores))


# In[ ]:


print ("train accuracy:", ovrm.score(X_train_tf, y_train_tf ))
print ("valid accuracy:", ovrm.score(X_valid_tf, y_valid_tf))


# In[ ]:


df_test.head()


# Creating separate dataframes for test predictions using logistic classifier and svc classifiers

# In[ ]:


df_test_logistic = df_test.copy()[["PhraseId"]]
df_test_logistic['Sentiment'] = ovrm.predict(test_df_flags)

df_test_logistic.head()


# Linear SVC

# SVC is similar to SVM classifier - but is more flexible with parameter tuning. Although, I didn't do much tuning myself. I don't believe that would get us very far. need to explore advanced models

# In[ ]:


svc = LinearSVC(dual=False)
svc.fit(X_train_tf, y_train_tf)


# In[ ]:


print ("train accuracy:", svc.score(X_train_tf, y_train_tf ))
print ("valid accuracy:", svc.score(X_valid_tf, y_valid_tf))


# In[ ]:


df_test_svm = df_test.copy()[["PhraseId"]]
df_test_svm['Sentiment'] = svc.predict(test_df_flags)
df_test_svm.head()


# Both models perform incredibly well on the training set but not as much on the validation set. If our test set is any similar to the training and validation sets, the accuracy shouldn't be much farther than the 50is %, if not less

# choosing to go with logistic regression predictions, for no particular reason. Don't expect to see much of a difference

# In[ ]:


df_test_logistic.to_csv("submission_tfidf_logistic.csv", index = False)


# In[ ]:




