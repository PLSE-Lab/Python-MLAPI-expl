#!/usr/bin/env python
# coding: utf-8

# I am trying to understand the kernel of Abhishek in Spooky Author Competition
# The Orginal Kernel is [Approaching (Almost) Any NLP Problem on Kaggle
# ](https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle). Therefore most of the code is from that Kernel
# 

# In[ ]:


import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from nltk import word_tokenize
from nltk.corpus import stopwords
import json

stop_words = stopwords.words('english')


# In[ ]:


train = pd.read_json('../input/whats-cooking-kernels-only/train.json')
test = pd.read_json('../input/whats-cooking-kernels-only/test.json')
sub = pd.read_csv('../input/whats-cooking-kernels-only/sample_submission.csv')


# In[ ]:


type(train)


# In[ ]:


train.head()


# In[ ]:


test.head()


# # Metric Code

# In[ ]:


def multiclass_logloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota


# We use the LabelEncoder from scikit-learn to convert text labels to integers, 0, 1 2

# In[ ]:


lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(train.cuisine.values)


# In[ ]:


y


# In[ ]:


type(train.ingredients)


# In[ ]:


train.ingredients.str.join(' ').head()


# Before going further it is important that we split the data into training and validation sets. We can do it using train_test_split from the model_selection module of scikit-learn.

# In[ ]:


xtrain, xvalid, ytrain, yvalid = train_test_split(train.ingredients.str.join(' '), y, 
                                                  stratify=y, 
                                                  random_state=42, 
                                                  test_size=0.1, shuffle=True)


# In[ ]:


xtrain.head()


# In[ ]:


ytrain


# In[ ]:


print (xtrain.shape)
print (xvalid.shape)


# # Building Basic Models

# ## TF -IDF and LR

# Let's start building our very first model.
# 
# Our very first model is a simple TF-IDF (Term Frequency - Inverse Document Frequency) followed by a simple Logistic Regression.

# In[ ]:


# Always start with these features. They work (almost) everytime!
# tfv = TfidfVectorizer(min_df=3,  max_features=None, 
#             strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
#             ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
#             stop_words = 'english')

tfv = TfidfVectorizer(max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            )

# Fitting TF-IDF to both training and test sets (semi-supervised learning)
tfv.fit(list(xtrain) + list(xvalid))
xtrain_tfv =  tfv.transform(xtrain) 
xvalid_tfv = tfv.transform(xvalid)


# In[ ]:


# Fitting a simple Logistic Regression on TFIDF
clf = LogisticRegression(C=1.0)
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict_proba(xvalid_tfv)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))


# ## Count Vectorizer and LR

# In[ ]:


ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), stop_words = 'english')

# Fitting Count Vectorizer to both training and test sets (semi-supervised learning)
ctv.fit(list(xtrain) + list(xvalid))
xtrain_ctv =  ctv.transform(xtrain) 
xvalid_ctv = ctv.transform(xvalid)


# In[ ]:


xtrain_ctv


# In[ ]:


# Fitting a simple Logistic Regression on Counts
clf = LogisticRegression(C=1.0)
clf.fit(xtrain_ctv, ytrain)
predictions = clf.predict_proba(xvalid_ctv)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))


# ## Multinomial Naive Bayes

# In[ ]:


# Fitting a simple Naive Bayes on TFIDF
clf = MultinomialNB()
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict_proba(xvalid_tfv)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))


# # Glove Vectors

# In[ ]:


# load the GloVe vectors in a dictionary:

embeddings_index = {}
f = open('../input/glove-global-vectors-for-word-representation/glove.6B.200d.txt')


# In[ ]:


f


# In[ ]:


for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# In[ ]:


# this function creates a normalized vector for the whole sentence
def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())


# In[ ]:


# create sentence vectors using the above function for training and validation set
xtrain_glove = [sent2vec(x) for x in tqdm(xtrain)]
xvalid_glove = [sent2vec(x) for x in tqdm(xvalid)]


# In[ ]:


xtrain_glove = np.array(xtrain_glove)
xvalid_glove = np.array(xvalid_glove)


# In[ ]:


xtrain_glove


# # XGBoost

# In[ ]:


# Fitting a simple xgboost on glove features
clf = xgb.XGBClassifier(nthread=10, silent=False)
clf.fit(xtrain_glove, ytrain)
predictions = clf.predict_proba(xvalid_glove)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))


# In[ ]:


# Fitting a simple xgboost on glove features
clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1, silent=False)
clf.fit(xtrain_glove, ytrain)
predictions = clf.predict_proba(xvalid_glove)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

