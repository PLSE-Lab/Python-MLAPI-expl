#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#changes:except LR all other models taking so much time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#import libraries
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sub = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


#shape of training dataset
train.shape


# In[ ]:


#shape of testing dataset
test.shape


# In[ ]:


#peek of the dataset
train.head()


# In[ ]:


#peek of the dataset
test.head()


# In[ ]:


#peek of the submission file
sub.head()


# In[ ]:


#check datatypes
train.dtypes


# In[ ]:


test.dtypes


# In[ ]:


import nltk.stem
from sklearn.feature_extraction.text import TfidfVectorizer
english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


# In[ ]:


import re
def clean_text( text ):
    # Function to convert a document to a sequence of words
    text = re.sub("[^A-za-z0-9^,?!.\/'+-=]"," ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " _exclamationmark_ ", text)
    text = re.sub(r"\?", " _questionmark_ ", text)
    return text


# In[ ]:


def build_data_set(ngram=3,stem=False,max_features=2000,min_df=2,remove_stopwords=True):
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    test.fillna('missing',inplace=True)
    clean_train_comments = []
    
    for i in range(train.shape[0]):
        clean_train_comments.append( clean_text(train["comment_text"][i]) )

    for i in range(test.shape[0]):
        clean_train_comments.append( clean_text(test["comment_text"][i]) )
        
    qs = pd.Series(clean_train_comments).astype(str)
    
    if not stem:
        # 1-gram / no-stem
        vect = TfidfVectorizer(analyzer=u'word',stop_words='english',
                               min_df=min_df,ngram_range=(1, ngram),max_features=max_features)
        ifidf_vect = vect.fit_transform(qs) 
        #print("ifidf_vect:", ifidf_vect.shape)
        X = ifidf_vect.toarray()
        X_train = X[:train.shape[0]]
        X_test = X[train.shape[0]:]
    else:
        vect_stem = StemmedTfidfVectorizer(analyzer=u'word',stop_words='english',
                                           min_df=min_df,ngram_range=(1, ngram),max_features=max_features)
        ifidf_vect_stem = vect_stem.fit_transform(qs)
        #print("ifidf_vect_stem:", ifidf_vect_stem.shape)
        X = ifidf_vect_stem.toarray()
        X_train = X[:train.shape[0]]
        X_test = X[train.shape[0]:]
    Y_train = train[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]
    assert Y_train.shape[0] == X_train.shape[0]
    del train, test
    return X_train,X_test,Y_train


# In[ ]:


labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
params = {
    'toxic': {'ngrams': 1, 'stem': True, 'max_features': 1000, 'C': 10 } , 
    'threat': {'ngrams': 1, 'stem': False, 'max_features': 1000, 'C': 10 } , 
    'severe_toxic': {'ngrams': 1, 'stem': True, 'max_features': 1000, 'C': 1.2 } , 
    'obscene': {'ngrams': 1, 'stem': True, 'max_features': 1000, 'C': 10 } , 
    'insult': {'ngrams': 1, 'stem': True, 'max_features': 1000, 'C': 1.2 } , 
    'identity_hate': {'ngrams': 1, 'stem': True, 'max_features': 1000, 'C': 10 } 
}


# In[ ]:


import time
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score


# In[ ]:


start_time = time.time()

for label in labels:
    print(">>> processing ",label)
    
    X_train,X_test,Y_train = build_data_set(ngram=params[label]['ngrams'],
                                            stem=params[label]['stem'],
                                            max_features=params[label]['max_features'],
                                            min_df=2,remove_stopwords=True)
    Y_train_lab = Y_train[label]
    seed = 7
    scoring = 'accuracy'
    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression()))
    #models.append(('LDA', LinearDiscriminantAnalysis()))
    #models.append(('KNN', KNeighborsClassifier()))
    #models.append(('CART', DecisionTreeClassifier()))
    #models.append(('NB', GaussianNB()))
    #models.append(('SVM', SVC()))
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train_lab, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)


# In[ ]:


#sub[label] = output
#sub.to_csv("output_.csv", index=False)

