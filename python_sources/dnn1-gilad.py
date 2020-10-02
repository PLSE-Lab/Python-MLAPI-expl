#!/usr/bin/env python
# coding: utf-8

# In[58]:


# my imports
from shutil import copyfile
import nltk
from nltk.corpus import stopwords, words
import os, sys, re, collections, string
from operator import itemgetter as at
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from gensim.models import Word2Vec
from sklearn.feature_extraction import text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, auc, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
import lightgbm as lgb
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from nltk.corpus import brown
from nltk.stem import SnowballStemmer
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
stop_words = set(stopwords.words('english')) 


# In[49]:


# lets Load the data:
JIGSAW_PATH = "../input/"
train = pd.read_csv(os.path.join(JIGSAW_PATH,'train.csv'), index_col='id')
test = pd.read_csv(os.path.join(JIGSAW_PATH,'test.csv'), index_col='id')


# In[67]:


train = train.dropna(subset = ['comment_text'], axis=0)
train.isna().sum()
train.shape


# In[50]:


class Preprocess_:
#     def __init__(self):
#         pass
    
    @staticmethod     
    def remove_html(given_text):
        """ remove html format from a string"""
        html_regex = r"<[^>]*>"
        return re.sub(html_regex," ",given_text)

    @staticmethod 
    def remove_punct(given_text):
        "remove punctuation from a string "
        punct_regex = r"[^\s\w<>]"
        return re.sub(punct_regex," ",given_text)

    @staticmethod 
    def remove_stopwords(given_text, stop_words):
#         stop = stopwords.words('english')
        return " ".join(x for x in given_text.split() if x not in stop_words)

    @staticmethod 
    def num_tokenizaion(given_text):
        num_regex = r"\d[\d\.\$]*"
        return re.sub(num_regex,"<NUM>",given_text)
    
    @staticmethod 
    def connect_very_with_next_word(given_text):
        my_regex = r"(?<=\svery)\s"
        return re.sub(my_regex,"_", given_text)
    
    @staticmethod 
    def get_last_sentense(given_text):
        return given_text.split(r'. [A-Z]')[-1]
    
    @staticmethod
    def reduce_lengthening(given_text):
        pattern = re.compile(r"(.)\1{2,}")
        return pattern.sub(r"\1\1", given_text)

    @staticmethod
    def correct_word_(given_text):
        return " ".join([difflib.get_close_matches(word, words.words(), n=1)[0]
                for word in given_text.split()])
    
    @staticmethod
    def lower_case(given_text):
        return " ".join([word.lower()
                for word in given_text.split()])     


# In[51]:


def fit_get_pr_curve(y_test, y_score, title_=''):
    print()
    preds = y_score[:, 0]
    precision, recall, thresholds = precision_recall_curve(y_test, preds)

    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title_)
    plt.show()
    return precision, recall, thresholds


# In[60]:


# cleanining the text:
def clean_text(text_series):
    clean_text_series = text_series.apply(lambda x: x.lower())    .apply(lambda x: Preprocess_.remove_punct(x))    .apply(lambda x: Preprocess_.num_tokenizaion(x))
    return clean_text_series


def tokenize(s):
    return text.sub(r' \1 ', s).split()


# In[68]:


train.comment_text = clean_text(train.comment_text)


# In[69]:


#let's build our first model - we will use a TFIDF vectorizer with 1 to 3 grams:
vectorizer = text.TfidfVectorizer(TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1))
vectorizer.fit(train.comment_text)
# BATCHSIZE = len()
# voc_ = {}
# for i in range(BATCHSIZE,len(ballanced_train),BATCHSIZE):
#     print(i)
#     vectorizer.fit(ballanced_train[(i-BATCHSIZE) : i].comment_text)
#     voc_ = {k: vectorizer.vocabulary_.get(k, 0)
#             + voc_.get(k, 0) 
#             for k in set(vectorizer.vocabulary_) 
#             | set(voc_) }


# In[70]:


X = vectorizer.transform(train.comment_text)


# In[71]:


model = LogisticRegression(C=1.0, dual=False, n_jobs=1)


# In[72]:


X_train, X_test, y_train, y_test = train_test_split(X,
                 [1 if val > 0.5 else 0 for val in train.target],
                 test_size=0.3,
                 random_state=42)


# In[73]:


model.fit(X_train, y_train)


# In[74]:


predicions_train = model.predict(X_train)
predicions_test = model.predict(X_test)


# In[75]:


print(classification_report(y_test, predicions_test))
# confusion_matrix()


# In[76]:


test.head()


# In[77]:


os.listdir(JIGSAW_PATH)


# In[78]:


df_submit = pd.read_csv(os.path.join(JIGSAW_PATH,'sample_submission.csv'), index_col='id')
df_submit.head()


# In[36]:


predictions = model.predict_proba(vectorizer.transform(clean_text(test.comment_text)))[:,1]
df_submit.prediction = predictions


# In[37]:


df_submit.to_csv(os.path.join('submission.csv'), index=True)
df_submit.head()


# In[38]:


df = pd.read_csv(os.path.join('submission.csv'))

