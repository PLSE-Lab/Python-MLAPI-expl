#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re

import nltk
import pandas as pd
import numpy as np
import unicodedata
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
from imblearn.over_sampling import SMOTE, SVMSMOTE
from sklearn.svm import LinearSVC
warnings.simplefilter("ignore")

from sklearn.model_selection import train_test_split
#from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import spacy
from spacy.lang.pt.examples import sentences
from sklearn.model_selection import StratifiedShuffleSplit


# In[ ]:


def strip_accents(text):

    try:
        text = unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3 
        pass

    text = unicodedata.normalize('NFD', text)           .encode('ascii', 'ignore')           .decode("utf-8")

    return str(text)


# In[ ]:


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stemmer = PorterStemmer()
    last_token = None
    output = []
    for token in tokens:
        if token != last_token:
            if token[0].islower():
                output.append(stemmer.stem(token))
            else:
                output.append(token)
        last_token = token
    return output


# In[ ]:


def clear_dataset(dataframe):
    # convert to lower case
    dataframe["title"] = dataframe['title'].str.lower()
    # remove accents
    dataframe["title"] = dataframe['title'].apply(strip_accents)
    # replace some abreviations
    dataframe["title"].replace(r"\sc/", " com ", regex=True, inplace=True)
    dataframe["title"].replace(r"\sc\s", " com ", regex=True, inplace=True)
    dataframe["title"].replace(r"\s*p/", " para ", regex=True, inplace=True)
    dataframe["title"].replace(r"\s*s/", " sem ", regex=True, inplace=True)
    dataframe["title"].replace(r"\stam\.", " tamanho ", regex=True, inplace=True)
    dataframe["title"].replace(r"\stam\s", " tamanho ", regex=True, inplace=True)
    # remove information that doesn't help the classifier
    dataframe["title"].replace(r"\ssem detalhe(s)?\s", " ", regex=True, inplace=True)
    dataframe["title"].replace(r"\simpecavel\s", " ", regex=True, inplace=True)
    dataframe["title"].replace(r"\sperfeito estado\s", " ", regex=True, inplace=True)
    dataframe["title"].replace(r"\spouco uso\s", " ", regex=True, inplace=True)
    dataframe["title"].replace(r"\s(nf-e)|(nfe)\s", " ", regex=True, inplace=True)
    # replace Xu -> X unidades for nothing
    dataframe["title"].replace(r"\s\d+\s?u\s", " ", regex=True, inplace=True)
    # replace any date with YEAR
    dataframe["title"].replace(r"\s(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})+(\s|$)", " YEAR ", regex=True, inplace=True)
    dataframe["title"].replace(r"\s\d{4}\s*", " YEAR ", regex=True,inplace=True)
    dataframe["title"].replace(r"\s\d{2}(\s|/)\d{2}\s", " YEAR ", regex=True,inplace=True)
    # replace synonyms
    dataframe["title"].replace(r'\s(homic\.|homicinetica)\s', " homocinetica ", regex=True,inplace=True)
    dataframe["title"].replace(r'\s(pc|notebook|desktop)\s', " computador ", regex=True,inplace=True)
    # replace some separators
    dataframe["title"].replace(r"[/+\-,*\.\(\)\[\]\{\}]", " ", regex=True,inplace=True)
    # replace digits with NUM
    dataframe["title"].replace(r"\s\d+\s", " NUM ", regex=True,inplace=True)
    dataframe["title"].replace(r"\s(NUM|YEAR)\s(a\s)?(NUM|YEAR)\s", " YEAR ", regex=True,inplace=True)
    dataframe["title"].replace(r"\d", "0", regex=True,inplace=True)
    # replace 2 or more spaces to one
    dataframe["title"].replace(r"\s{2,}", " ", regex=True,inplace=True)


# In[ ]:


train_df = pd.read_csv('/kaggle/input/nlp-deeplearningbrasil-ml-challenge/train.csv')
clear_dataset(train_df)


# In[ ]:


# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(train_df[['title']], train_df.category, random_state=42, stratify=train_df.category)


# In[ ]:


stop_words = nltk.corpus.stopwords.words('portuguese')
# NLP Pipeline
text_clf_linearSVC  = Pipeline([
                # Vectorize
        
#                ('vect',  CountVectorizer(tokenizer=tokenize, 
#                                          stop_words=stop_words, 
#                                          ngram_range=(1,2),
#                                          max_df=0.1)),
    
                ('vect',  TfidfVectorizer(tokenizer=tokenize, 
                                            stop_words=stop_words, 
                                            ngram_range=(1,2))),


#                ('smoth', SMOTE(ratio='minority')),
#                 ('smoth', SVMSMOTE(sampling_strategy='minority')),
    
#                 ('vect',  HashingVectorizer(tokenizer=tokenize, 
#                                           stop_words=stop_words, 
#                                           ngram_range=(1,2))),
    
                ('clf', LinearSVC()),
            ])

text_clf_SGD   = Pipeline([
                # Vectorize
        
#                ('vect',  CountVectorizer(tokenizer=tokenize, 
#                                          stop_words=stop_words, 
#                                          ngram_range=(1,2),
#                                          max_df=0.1)),
    
                ('vect',  TfidfVectorizer(tokenizer=tokenize, 
                                            stop_words=stop_words, 
                                            ngram_range=(1,2))),


#                ('smoth', SMOTE(ratio='minority')),
#                 ('smoth', SVMSMOTE(sampling_strategy='minority')),
    
#                 ('vect',  HashingVectorizer(tokenizer=tokenize, 
#                                           stop_words=stop_words, 
#                                           ngram_range=(1,2))),
    
                # Classificador
                ('clf',   SGDClassifier(tol=1e-5,
                                     loss='modified_huber')),
#                ('clf', LinearSVC()),
            ])


# In[ ]:


# Train linearSVC
text_clf_linearSVC = text_clf_linearSVC.fit(X_train.title, y_train)
score_svc = text_clf_linearSVC.score(X_test.title, y_test)
print(score_svc)


# In[ ]:


# Train SGD
text_clf_SGD = text_clf_SGD.fit(X_train.title, y_train)
score_SGD = text_clf_SGD.score(X_test.title, y_test)
print(score_SGD)


# In[ ]:


# Choose classifier
if score_SGD > score_svc:
    text_clf = text_clf_SGD
else:
    text_clf = text_clf_linearSVC


# In[ ]:


# Evaluate
predictions = text_clf.predict(X_test.title)
print(metrics.classification_report(y_test, predictions, target_names=train_df['category'].unique()))


# In[ ]:


# Retrain with the entire base
X_train, y_train = train_df[['title']], train_df[['category']]
text_clf = text_clf.fit(X_train.title, y_train)


# In[ ]:


# Predict
df = pd.read_csv('/kaggle/input/nlp-deeplearningbrasil-ml-challenge/test.csv')
clear_dataset(df)
res = text_clf.predict(df.title)
with open("to_submit.csv", 'w') as f:
    cnt = 0
    f.write('id,category\n')
    for cat in res:
        f.write("{0},{1}\n".format(cnt, cat))
        cnt = cnt + 1;


# In[ ]:




