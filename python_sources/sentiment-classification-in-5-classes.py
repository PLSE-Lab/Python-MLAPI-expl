#!/usr/bin/env python
# coding: utf-8

# ## Predictive Analysis##
# We are trying to predict the sentiment of the comments recorded with people into 5 classes. The idea is to use POS tagging in order to improve the accuracy of the model.
# 
# *Note :- The code is published with a dataset of 25000 comments only because of kernel idle time*
# 
# 
# ## Loading data into pandas dataframe ##
# We have loaded data from csv into pandas dataframe , have dropped non useful columns from the dataframe and have only kept "Text" and "Score" which is useful to us.

# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

import pandas as pd
import numpy as np

np.random.seed(0)

def read_text_file(f):
    df_complete = pd.read_csv(f)
    df = df_complete.loc[:,["Text","Score"]]
    df.dropna(how="any", inplace=True)    
    return df

df = read_text_file("../input/Reviews.csv")
print (df.head())


# ##Sampling Data
# Sampling dataset to bring each of the classes into same frequency

# In[ ]:


def sampling_dataset(df):
    count = 5000
    class_df_sampled = pd.DataFrame(columns = ["Score","Text"])
    temp = []
    for c in df.Score.unique():
        class_indexes = df[df.Score == c].index
        random_indexes = np.random.choice(class_indexes, count, replace=False)
        temp.append(df.loc[random_indexes])
        
    for each_df in temp:
        class_df_sampled = pd.concat([class_df_sampled,each_df],axis=0)
    
    return class_df_sampled

df = sampling_dataset(df)
df.reset_index(drop=True,inplace=True)
print (df.head())
print (df.shape)


# ## Cleaning data to remove Stopwords and Small length words also lemmatized data to bring into common format ##
# 
# We have used NLTK library to tokenize words , remove stopwords and lemmatize the remaining words. Also the part of speech for each of the word is added to the word to counteract cases where POS varies meaning of the word.

# In[ ]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
import re

lmtzr = WordNetLemmatizer()
negation = re.compile(r"(?:^(?:never|no|nothing|nowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint)$)|n't",re.I)
clp = re.compile(r"^[.:;!?]$",re.I)
    
def extract_words_from_comments(df):
    comments_tok = []
    for index, datapoint in df.iterrows():
        tokenized_words = word_tokenize(datapoint["Text"].lower(),language='english')
        pos_tagged_words = pos_tag(tokenized_words)
        tokenized_words = ["_".join([lmtzr.lemmatize(i[0]),i[1]]) for i in pos_tagged_words if (i[0] not in stopwords.words("english") and len(i[0]) > 2)]
        comments_tok.append(tokenized_words)
    df["comment_tok"] = comments_tok
    return df

df = extract_words_from_comments(df)
print (df.head())
print (df.shape)


# ## Vectorize words using BOW technique ##
# We have used gensim to create library and convert tokenized sentences into vectors using BOW technique

# In[ ]:


from gensim import matutils,corpora, models

def vectorize_comments(df):
    d = corpora.Dictionary(df["comment_tok"])
    d.filter_extremes(no_below=2, no_above=0.8)
    d.compactify()
    corpus = [d.doc2bow(text) for text in df["comment_tok"]]
    # tfidf = TfidfModel(corpus=corpus,id2word=d)
    # corpus_tfidf = tfidf[corpus]
    # corpus_tfidf = matutils.corpus2csc(corpus_tfidf,num_terms=len(d.token2id))
    corpus = matutils.corpus2csc(corpus, num_terms=len(d.token2id))
    corpus = corpus.transpose()
    return d, corpus

dictionary,corpus = vectorize_comments(df)
print (corpus.shape)


# ## Train Random forest classifier on a grid search parameters ##
# Training RFC classifier with grid search to tune hyperparameters . Score on test data and highest CV score  printed.

# In[ ]:


from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier as RFC
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pickle

def train_classifier(X,y):
    n_estimators = [100]
    min_samples_split = [2]
    min_samples_leaf = [1]
    bootstrap = [True]

    parameters = {'n_estimators': n_estimators, 'min_samples_leaf': min_samples_leaf,
                  'min_samples_split': min_samples_split}

    clf = GridSearchCV(RFC(verbose=1,n_jobs=4), cv=4, param_grid=parameters)
    clf.fit(X, y)
    return clf

X_train, X_test, y_train, y_test = cross_validation.train_test_split(corpus, df["Score"], test_size=0.02, random_state=17)
classifier = train_classifier(X_train,y_train)
print (classifier.best_score_, "----------------Best Accuracy score on Cross Validation Sets")
print (classifier.score(X_test,y_test))


# In[ ]:


f = open("Output.txt","w")
f.write("Best Accuracy score on Cross Validation Sets %f" %classifier.best_score_,)
f.write("Score on Test Set %f" %classifier.score(X_test,y_test))
f.close()

