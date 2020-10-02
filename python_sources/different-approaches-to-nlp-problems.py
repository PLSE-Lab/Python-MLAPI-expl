#!/usr/bin/env python
# coding: utf-8

# # Different Approaches To NLP Problems
# 
# <img src="https://s3.amazonaws.com/codecademy-content/courses/NLP/Natural_Language_Processing_Overview.gif" height="500" width="500">
# [Image Source](https://s3.amazonaws.com/codecademy-content/courses/NLP/Natural_Language_Processing_Overview.gif)
# 
# # CONTEXT 
# * [Importing packages and Reading Data](#1)
# * [Data Preprocessing](#2)
#     * [Text Cleaning](#2.1)
#     * [Tokenizer](#2.2)
#     * [Remove StopWord](#2.3)
#     * [Token normalization](#2.4)
#     * [Transforming tokens to a vector](#2.5)
# * [Building Model](#3)
#     * [Logistic Regression](#3.1)
#     * [SVC](#3.2)
#     * [MultinomialNB](#3.3)
#     * [XGBoost](#3.4)
# * [Grid Search](#4)
# * [GloVe](#5)
# * [Deep Learning](#6)
#     * [Sequential Neural Net](#6.1) 
#     * [LSTM](#6.2)
#     * [GRU](#6.3)
# * [References](#7)
# 
# 
# # Intro
# In this notebook, I just want to try different approaches, methods and modeles to solve NLP problems. We are going to start with very basic model and feature engineering and then improve it using different other models. We will also use deep neural networks and see how its perform compare to others. Last but not least we will try emsembling.
# 
# # Data
# **Each sample in the train and test set has the following information:**
# 
# * The text of a tweet
# * A keyword from that tweet (although this may be blank!)
# * The location the tweet was sent from (may also be blank)
# 
# # What am I predicting?
# You are predicting whether a given tweet is about a real disaster or not. If so, predict a 1. If not, predict a 0.
# 
# **Files**
# * train.csv - the training set
# * test.csv - the test set
# * sample_submission.csv - a sample submission file in the correct format
# **Columns**
# * id - a unique identifier for each tweet
# * text - the text of the tweet
# * location - the location the tweet was sent from (may be blank)
# * keyword - a particular keyword from the tweet (may be blank)
# * target - in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)

# # 1. Importing packages and Reading Data <a id="1"></a>

# In[ ]:


"""Importing libraries"""
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.metrics import f1_score

from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping


# In[ ]:


"""Let's load the data files"""
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
sub = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')


# In[ ]:


"""Reading train data"""
print(train.shape)
train.head()


# In[ ]:


"""Reading test data"""
print(test.shape)
test.head()


# In[ ]:


"""reading submission file"""
sub.head()


# # 2. Data Preprocessing <a id="2"></a>

# In[ ]:


xtrain, xvalid, ytrain, yvalid = train_test_split(train.text, 
                                                  train.target,
                                                  random_state=42, 
                                                  test_size=0.1, shuffle=True)


# In[ ]:


print(xtrain.shape)
print(xvalid.shape)


# ##  2.1 Text Cleaning <a id="2.1"></a>

# In[ ]:


get_ipython().run_cell_magic('time', '', "def clean_text(text):\n    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation\n    and remove words containing numbers.'''\n    text = text.lower()\n    text = re.sub('\\[.*?\\]', '', text)\n    text = re.sub('https?://\\S+|www\\.\\S+', '', text)\n    text = re.sub('<.*?>+', '', text)\n    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)\n    text = re.sub('\\n', '', text)\n    text = re.sub('\\w*\\d\\w*', '', text)\n    return text\n\n\n# Applying the cleaning function to both test and training datasets\nxtrain = xtrain.apply(lambda x: clean_text(x))\nxvalid = xvalid.apply(lambda x: clean_text(x))\nxtrain.head(3)")


# ## 2.2 Tokenizer <a id="2.2"></a>
# [Documentation](https://www.nltk.org/api/nltk.tokenize.html****)

# In[ ]:


get_ipython().run_cell_magic('time', '', "tokenizer1 = nltk.tokenize.WhitespaceTokenizer()\ntokenizer2 = nltk.tokenize.TreebankWordTokenizer()\ntokenizer3 = nltk.tokenize.WordPunctTokenizer()\ntokenizer4 = nltk.tokenize.RegexpTokenizer(r'\\w+')\ntokenizer5 = nltk.tokenize.TweetTokenizer()\n\n# appling tokenizer5\nxtrain = xtrain.apply(lambda x: tokenizer5.tokenize(x))\nxvalid = xvalid.apply(lambda x: tokenizer5.tokenize(x))\nxtrain.head(3)")


# ## 2.3 Remove StopWord <a id="2.3"></a>
# [Documentation](https://www.geeksforgeeks.org/removing-stop-words-nltk-python/)
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'def remove_stopwords(text):\n    """\n    Removing stopwords belonging to english language\n    \n    """\n    words = [w for w in text if w not in stopwords.words(\'english\')]\n    return words\n\n\nxtrain = xtrain.apply(lambda x : remove_stopwords(x))\nxvalid = xvalid.apply(lambda x : remove_stopwords(x))')


# In[ ]:


get_ipython().run_cell_magic('time', '', "def combine_text(list_of_text):\n    combined_text = ' '.join(list_of_text)\n    return combined_text\n\nxtrain = xtrain.apply(lambda x : combine_text(x))\nxvalid = xvalid.apply(lambda x : combine_text(x))")


# ## 2.4 Token normalization <a id="2.4"></a>
# Token normalisation means converting different tokens to their base forms. This can be done either by:
# 
# * Stemming : removing and replacing suffixes to get to the root form of the word, which is called the stem for instance cats - cat, wolves - wolv
# * Lemmatization : Returns the base or dictionary form of a word, which is known as the lemma
# 
# [source](https://www.google.com/search?q=not+least+but+last&oq=not+least&aqs=chrome.7.69i57j0l7.22351j0j1&sourceid=chrome&ie=UTF-8)

# In[ ]:


# Stemmer
stemmer = nltk.stem.PorterStemmer()

# Lemmatizer
lemmatizer=nltk.stem.WordNetLemmatizer()

# Appling Lemmatizer
xtrain = xtrain.apply(lambda x: lemmatizer.lemmatize(x))
xvalid = xvalid.apply(lambda x: lemmatizer.lemmatize(x))


# ## 2.5 Transforming tokens to a vector <a id="2.5"></a>
# [Countvectorizer Features](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
# 
# [TFIDF Features](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
# 
# *[Reading](http://www.tfidf.com/)*

# In[ ]:


# Appling CountVectorizer()
count_vectorizer = CountVectorizer()
xtrain_vectors = count_vectorizer.fit_transform(xtrain)
xvalid_vectors = count_vectorizer.transform(xvalid)


# In[ ]:


# Appling TFIDF
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2), norm='l2')
xtrain_tfidf = tfidf.fit_transform(xtrain)
xvalid_tfidf = tfidf.transform(xvalid)


# # 3.Building Models <a id="3"></a>
# 
# ## Logistic Regression <a id="3.1"></a>

# In[ ]:


# Fitting a simple Logistic Regression on TFIDF
clf = LogisticRegression(C=1.0)
clf.fit(xtrain_tfidf, ytrain)
#scores = model_selection.cross_val_score(clf, train_tfidf, ytrain, cv=5, scoring="f1")

predictions = clf.predict(xvalid_tfidf)
print('simple Logistic Regression on TFIDF')
print ("f1_score :", np.round(f1_score(yvalid, predictions),5))


# In[ ]:


# Fitting a simple Logistic Regression on CountVec
clf = LogisticRegression(C=1.0)
clf.fit(xtrain_vectors, ytrain)
#scores = model_selection.cross_val_score(clf, xtrain_vectors, ytrain_vectors, cv=5, scoring="f1")

predictions = clf.predict(xvalid_vectors)
print('simple Logistic Regression on CountVectorizer')
print ("f1_score :", np.round(f1_score(yvalid, predictions),5))


# Hmm!! We just improved our first model by 0.01.
# 
# But, we can find different scores by playing with the parameters of count, tfidf and model.
# 
# [about f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
# 
# Let's try SVC and Naives Bayes Classifier
# 
# ## SVC <a id="3.2"></a>

# In[ ]:


# Fitting a LinearSVC on TFIDF
clf = SVC()
clf.fit(xtrain_tfidf, ytrain)

predictions = clf.predict(xvalid_tfidf)
print('SVC on TFIDF')
print ("f1_score :", np.round(f1_score(yvalid, predictions),5))


# In[ ]:


# Fitting a LinearSVC on CountVec
clf = SVC()
clf.fit(xtrain_vectors, ytrain)

predictions = clf.predict(xvalid_vectors)
print('SVC on CountVectorizer')
print ("f1_score :", np.round(f1_score(yvalid, predictions),5))


# Bad performance by svc on this dataset!!
# 
# ## MultinomialNB <a id="3.3"></a>

# In[ ]:


# Fitting a MultinomialNB on TFIDF
clf = MultinomialNB()
clf.fit(xtrain_tfidf, ytrain)

predictions = clf.predict(xvalid_tfidf)
print('MultinomialNB on TFIDF')
print ("f1_score :", np.round(f1_score(yvalid, predictions),5))


# In[ ]:


# Fitting a MultinomialNB on CountVec
clf = MultinomialNB()
clf.fit(xtrain_vectors, ytrain)

predictions = clf.predict(xvalid_vectors)
print('MultinomialNB on CountVectorizer')
print ("f1_score :", np.round(f1_score(yvalid, predictions),5))


# Good!! but the logistic regression on counts is still better than other ancient models we try here. Let's jump into XGBoost model.
# 
# ## XGBoost <a id="3.4"></a>
# 
# [Parameters](https://xgboost.readthedocs.io/en/latest/parameter.html)

# In[ ]:


# Fitting a simple xgboost on TFIDF
clf = xgb.XGBClassifier(max_depth=5, n_estimators=300, colsample_bytree=0.8, 
                        subsample=0.5, nthread=10, learning_rate=0.1)
clf.fit(xtrain_tfidf.tocsc(), ytrain)
predictions = clf.predict(xvalid_tfidf.tocsc())

print('XGBClassifier on TFIDF')
print ("f1_score :", np.round(f1_score(yvalid, predictions),5))


# In[ ]:


# Fitting a simple xgboost on CountVec
clf = xgb.XGBClassifier(max_depth=5, n_estimators=300, colsample_bytree=0.8, 
                        subsample=0.5, nthread=10, learning_rate=0.1)
clf.fit(xtrain_vectors, ytrain)

predictions = clf.predict(xvalid_vectors)
print('XGBClassifier on CountVectorizer')
print ("f1_score :", np.round(f1_score(yvalid, predictions),5))


# It seens like XGBoost perform worst than other! but that is not correct. I haven't done any hyperparameter optimizations yet. Let's do it.
# 
# # Grid Search <a id="4"></a>
# Now let's add Grid Search to all the models with the hopes of optimizing their hyperparameters and thus improving their accuracy. Are the default model parameters the best bet? Let's find out.

# In[ ]:


'''Create a function to tune hyperparameters of the selected models.'''
seed = 44
def grid_search_cv(model, params):
    global best_params, best_score
    grid_search = GridSearchCV(estimator = model, param_grid = params, cv = 5, 
                             verbose = 3,
                             scoring = 'f1', n_jobs = -1)
    grid_search.fit(xtrain_vectors, ytrain)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    return best_params, best_score


# In[ ]:


'''Define hyperparameters of Logistic Regression.'''
LR_model = LogisticRegression()

LR_params = {'penalty':['l1', 'l2'],
             'C': np.logspace(0.1, 1, 4, 8 ,10)}

grid_search_cv(LR_model, LR_params)
LR_best_params, LR_best_score = best_params, best_score
print('LR best params:{} & best_score:{:0.5f}' .format(LR_best_params, LR_best_score))


# In[ ]:


'''Define hyperparameters of Logistic Regression.'''
SVC_model = SVC()

SVC_params = {'kernel':[ 'linear', 'rbf', 'sigmoid'],
             'C': np.logspace(0.1, 1,10)}

grid_search_cv(SVC_model, SVC_params)
SVC_best_params, SVC_best_score = best_params, best_score
print('SVC best params:{} & best_score:{:0.5f}' .format(SVC_best_params, SVC_best_score))


# In[ ]:


'''Define hyperparameters of Logistic Regression.'''
NB_model = MultinomialNB()

NB_params = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

grid_search_cv(NB_model, NB_params)
NB_best_params, NB_best_score = best_params, best_score
print('NB best params:{} & best_score:{:0.5f}' .format(NB_best_params, NB_best_score))


# After hyperparameter tuning, we can see that countvector data give us improved result. Before our best score was **0.73515 (LR)** and now after gridsearch our score is **0.75783 (NB)**.
# 
# I am not optimizing XGBoost hyperparameters becuase its  is time consuming. but you can try optimization on all the models for better score.

# In[ ]:


#'''For XGBC, the following hyperparameters are usually tunned.'''
#'''https://xgboost.readthedocs.io/en/latest/parameter.html'''

#XGB_model = XGBClassifier(
#            n_estimators=500,
#            verbose = True)


#XGB_params = {'max_depth': (2, 5),
#               'reg_alpha':  (0.01, 0.4),
#               'reg_lambda': (0.01, 0.4),
#               'learning_rate': (0.1, 0.4),
#               'colsample_bytree': (0.3, 1),
#               'gamma': (0.01, 0.7),
#               'num_leaves': (2, 5),
#               'min_child_samples': (1, 5),
#              'subsample': [0.5, 0.8],
#              'random_state':[seed]}

#grid_search_cv(XGB_model, XGB_params)
#XGB_best_params, XGB_best_score = best_params, best_score
#print('XGB best params:{} & best_score:{:0.5f}' .format(XGB_best_params, XGB_best_score))


# # GloVe for Vectorization <a id="5"></a>
# Here we will use GloVe pretrained corpus model to represent our words.It is available in 3 varieties :50D ,100D and 200 Dimentional.We will try 200 D here.

# In[ ]:


"""Load the Glove vectors in a dictionay"""
embeddings_index={}
with open('../input/glove-global-vectors-for-word-representation/glove.6B.200d.txt','r') as f:
    for line in f:
        values=line.split()
        word=values[0]
        vectors=np.asarray(values[1:],'float32')
        embeddings_index[word]=vectors
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# In[ ]:


""" Function Creates a normalized vector for the whole sentence"""
def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stopwords.words('english')]
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
        return np.zeros(200)
    return v / np.sqrt((v ** 2).sum())


# In[ ]:


# create sentence vectors using the above function for training and validation set
# create glove features
xtrain_glove = np.array([sent2vec(x) for x in tqdm(xtrain)])
xvalid_glove = np.array([sent2vec(x) for x in tqdm(xvalid)])


# In[ ]:


# Shape of data after embedding
xtrain_glove.shape,  xvalid_glove.shape


# In[ ]:


# Fitting a simple xgboost on glove features
clf = xgb.XGBClassifier(max_depth=8, n_estimators=300, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1)
clf.fit(xtrain_glove, ytrain)

predictions = clf.predict(xvalid_glove)
print('XGBClassifier on GloVe featur')
print ("f1_score :", np.round(f1_score(yvalid, predictions),5))


# When we compare the previous xgboost results with XGBClassifier on GloVe feature result we can see that score has been increase by 0.06. we can further improve by tuning of parameters.
# 
# # Deep Learning <a id="6"></a>
# 
# ## Sequential Neural Net <a id="6.1"></a>

# In[ ]:


"""scale the data before any neural net"""
scl = preprocessing.StandardScaler()
xtrain_glove_scl = scl.fit_transform(xtrain_glove)
xvalid_glove_scl = scl.transform(xvalid_glove)


# In[ ]:


"""create a simple 2 layer sequential neural net"""
model = Sequential()

model.add(Dense(200, input_dim=200, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(1))
model.add(Activation('sigmoid'))

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


model.fit(xtrain_glove_scl, y=ytrain, batch_size=64, 
          epochs=10, verbose=1, 
          validation_data=(xvalid_glove_scl, yvalid))


# In[ ]:


predictions = model.predict(xvalid_glove_scl)
predictions = np.round(predictions).astype(int)
print('2 layer sequential neural net on GloVe Feature')
print ("f1_score :", np.round(f1_score(yvalid, predictions),5))


# Nice!! 2 layer sequential neural net on GloVe Feature perform better than xgboost

# ## LSTM <a id="6.2"></a>
# For LSTM modeling we need to tokensize the text data:

# In[ ]:


# using keras tokenizer here
token = text.Tokenizer(num_words=None)
max_len = 80

token.fit_on_texts(list(xtrain) + list(xvalid))
xtrain_seq = token.texts_to_sequences(xtrain)
xvalid_seq = token.texts_to_sequences(xvalid)

# zero pad the sequences
xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)

word_index = token.word_index
print('Number of unique words:',len(word_index))


# In[ ]:


#create an embedding matrix for the words we have in the dataset
embedding_matrix = np.zeros((len(word_index) + 1, 200))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# In[ ]:


# A simple LSTM with glove embeddings and two dense layers
model = Sequential()
model.add(Embedding(len(word_index) + 1,
                     200,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
model.add(SpatialDropout1D(0.3))
model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))

model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


# Fit the model with early stopping callback
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')

model.fit(xtrain_pad, y=ytrain, batch_size=512, epochs=100, verbose=1, validation_data=(xvalid_pad, yvalid), callbacks=[earlystop])


# In[ ]:


predictions = model.predict(xvalid_pad)
predictions = np.round(predictions).astype(int)

print('simple LSTM')
print ("f1_score :", np.round(f1_score(yvalid, predictions),5))


# Waoh!! LSTM model perform as expected. Best score till now!!.
# 
# ## GRU <a id="6.3"></a>

# In[ ]:


# A simple LSTM with glove embeddings and two dense layers
model = Sequential()
model.add(Embedding(len(word_index) + 1,
                     200,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
model.add(SpatialDropout1D(0.3))
model.add(GRU(100, dropout=0.3, recurrent_dropout=0.3))

model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


# Fit the model with early stopping callback
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')

model.fit(xtrain_pad, y=ytrain, batch_size=512, epochs=100, verbose=1, validation_data=(xvalid_pad, yvalid), callbacks=[earlystop])


# In[ ]:


predictions = model.predict(xvalid_pad)
predictions = np.round(predictions).astype(int)
print('simple GRU')
print ("f1_score :", np.round(f1_score(yvalid, predictions),5))


# # References <a id="7"></a>
# 1. https://www.kaggle.com/vikassingh1996/simple-model-feat-nlp-disaster-tweets-lb-0-80572
# 2. https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove
# 3. https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle
# 
# ## Give me your feedback and if you find my kernel helpful please UPVOTE will be appreciated
