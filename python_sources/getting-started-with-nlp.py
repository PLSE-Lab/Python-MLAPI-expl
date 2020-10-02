#!/usr/bin/env python
# coding: utf-8

# ### This notebook is completely inspired from [Approaching Any Problem of NLP](https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle)
# 
# All the steps covered in that notebook will be replicated here with changes and meaning to each and every step we take.
# 
# This notebook is for all those like me, who want to get started with NLP
# 
# ### The book is going to Cover the following points
# * TFIDF
# * Count Features
# * Logistic Regression
# *  Naive Bais
# * SVM
# * xgboost
# * GridSearch
# * word vectors
# * LSTM
# * GRU
# * Ensembling
# 
# We'll cover all these points and let's have our dataset from Spooky Author Identification
# 
# ### Imports

# In[ ]:


import pandas as pd
import numpy as np
import xgboost as xgb
import tqdm as tqdm #for progress bars
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers.recurrent import GRU, LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils # array and list manipulation 
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
#The purpose of the pipeline is to assemble several steps that can be cross-validated 
#together while setting different parameters.
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD #Dimensionality reduction
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB #suitable for classification with discrete features 
from keras.layers import Bidirectional, GlobalAvgPool1D, Conv1D, Flatten, SpatialDropout1D, MaxPooling1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


# ### Let's load the dataset

# In[ ]:


train = pd.read_csv('../input/spooky-author-identification/train.csv')
test = pd.read_csv('../input/spooky-author-identification/test.csv')


# In[ ]:


train.head(5)


# ### What is the problem and what are we expected to do ?
# So the problem given is 'Given a text sentence we have to identifiy which author might have written it'
# 
# As described in the Overview, 3 Authors are there EAP, HPL and MWS so basically we have the task of classifiying between 3 authors - text classification between 3 classes
# 
# ### Metric
# 
# For this problem we have given metric as multiclass log loss as our evaluation metric  - http://wiki.fast.ai/index.php/Log_Loss#Multi-class_Classification
# 
#  (taken from: https://github.com/dnouri/nolearn/blob/master/nolearn/lasagne/util.py)

# In[ ]:


def multiclass_logloss(actual, predicted, eps = 1e-15):
    #We have to make sure actual is binary 
    if(len(actual.shape) == 1):
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2
    clip = np.clip(predicted, eps, 1-eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0/rows * vsota


# We'll use label encoder to convert text to label -> 'a','b','c' = 0,1,2 etc
# 
# Here we'll convert our authors to integer
# 
# EAP,HPL,MWS => 0,1,2

# In[ ]:


lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(train.author.values)
set(y)


# In[ ]:


xtrain, xvalid, ytrain, yvalid = train_test_split(train.text.values, y, stratify = y, random_state=42, test_size=0.1, shuffle=True)


# In[ ]:


print(xtrain.shape, xvalid.shape)


# ### Let's try out basic model
# Let's do tfifd followed by Logistic Regression

# In[ ]:


tfv = TfidfVectorizer(min_df = 3, max_features=None, strip_accents='unicode', analyzer='word', token_pattern = r'\w{1,}',
                     ngram_range=(1,3), use_idf = 1, smooth_idf = 1, sublinear_tf = 1, stop_words = 'english')


# In[ ]:


# Fitting TF-IDF to both training and test sets (semi-supervised learning)
tfv.fit(list(xtrain) + list(xvalid))
xtrain_tfv = tfv.transform(xtrain)
xvalid_tfv = tfv.transform(xvalid)


# In[ ]:


#so basically it coverts your words into int and also give it's frequencies along with it's IDF
for x in xtrain_tfv:
    print(x)
    break


# In[ ]:


clf = LogisticRegression(C=1.0)
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict_proba(xvalid_tfv)
print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))


# And there we go. We have our first model with a multiclass logloss of 0.626.
# 
# Now let's use wordcount as a feature, this can be done by countvectorizer 

# In[ ]:


ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), stop_words = 'english')
ctv.fit(list(xtrain) + list(xvalid))
xtrain_ctv = ctv.transform(xtrain)
xvalid_ctv = ctv.transform(xvalid)


# In[ ]:


clf = LogisticRegression(C=1.0)
clf.fit(xtrain_ctv, ytrain)
predictions = clf.predict_proba(xvalid_ctv)
print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))


# We imporved 0.1 that's a significant change
# 
# Now let's try out naive bias
# 
# On tfidf

# In[ ]:


clf = MultinomialNB()
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict_proba(xvalid_tfv)
print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))


# In[ ]:


#On count vector
clf = MultinomialNB()
clf.fit(xtrain_ctv, ytrain)
predictions = clf.predict_proba(xvalid_ctv)
print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))


# That's a very good score
# 
# Now let's try SVM
# 
# As SVM takes a lot of time so let's reduce the number of features using  Singular Value Decomposition 

# In[ ]:


# Apply SVD, I chose 120 components. 120-200 components are good enough for SVM model.
svd = decomposition.TruncatedSVD(n_components=120)
svd.fit(xtrain_tfv)
xtrain_svd = svd.transform(xtrain_tfv)
xvalid_svd = svd.transform(xvalid_tfv)


# In[ ]:


# Scale the data obtained from SVD. Renaming variable to reuse without scaling.
scl = preprocessing.StandardScaler()
scl.fit(xtrain_svd)
xtrain_svd_scl = scl.transform(xtrain_svd)
xvalid_svd_scl = scl.transform(xvalid_svd)


# In[ ]:


clf = SVC(C=1.0, probability=True)
clf.fit(xtrain_svd_scl, ytrain)
predictions = clf.predict_proba(xvalid_svd_scl)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))


# Does't do we'll
# 
# Let's apply xgboost

# In[ ]:


# Fitting a simple xgboost on tf-idf
clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1)
clf.fit(xtrain_tfv.tocsc(), ytrain)
predictions = clf.predict_proba(xvalid_tfv.tocsc())

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

# Fitting a simple xgboost on tf-idf
clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1)
clf.fit(xtrain_ctv.tocsc(), ytrain)
predictions = clf.predict_proba(xvalid_ctv.tocsc())

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

# Fitting a simple xgboost on tf-idf svd features
clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1)
clf.fit(xtrain_svd, ytrain)
predictions = clf.predict_proba(xvalid_svd)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

# Fitting a simple xgboost on tf-idf svd features
clf = xgb.XGBClassifier(nthread=10)
clf.fit(xtrain_svd, ytrain)
predictions = clf.predict_proba(xvalid_svd)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))


# It does't do we'll 

# ## GridSearch
# 
# Let's try to search the best hyperparameter for best optimization Grid search using logistic regression.
# 
# Before starting with grid search we need to create a scoring function. This is accomplished using the make_scorer function of scikit-learn.

# In[ ]:


mll_scorer = metrics.make_scorer(multiclass_logloss, greater_is_better=False, needs_proba=True)


# In[ ]:


# Initialize SVD
svd = TruncatedSVD()
    
# Initialize the standard scaler 
scl = preprocessing.StandardScaler()

# We will use logistic regression here..
lr_model = LogisticRegression()

# Create the pipeline 
clf = pipeline.Pipeline([('svd', svd),
                         ('scl', scl),
                         ('lr', lr_model)])


# In[ ]:


param_grid = {'svd__n_components' : [120, 180],
              'lr__C': [0.1, 1.0, 10], 
              'lr__penalty': ['l1', 'l2']}


# In[ ]:


model = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=mll_scorer, verbose=10, iid = True, n_jobs=-1, refit=True, cv = 2)
model.fit(xtrain_tfv, ytrain)
print("Best score: %0.3f" % model.best_score_)
print("Best parameters set:")
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


# The score comes similar to what we had for SVM. This technique can be used to finetune xgboost or even multinomial naive bayes as below. We will use the tfidf data here

# In[ ]:


nb_model = MultinomialNB()

# Create the pipeline 
clf = pipeline.Pipeline([('nb', nb_model)])

# parameter grid
param_grid = {'nb__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

# Initialize Grid Search Model
model = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=mll_scorer,
                                 verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)

# Fit Grid Search Model
model.fit(xtrain_tfv, ytrain)  # we can use the full data here but im only using xtrain. 
print("Best score: %0.3f" % model.best_score_)
print("Best parameters set:")
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


# In NLP problems, it's customary to look at word vectors. Word vectors give a lot of insights about the data. Let's dive into that.
# 

# ## Wrod Vector
# Without going into too much details, I would explain how to create sentence vectors and how can we use them to create a machine learning model on top of it. I am a fan of GloVe vectors, word2vec and fasttext. In this post, I'll be using the GloVe vectors. You can download the GloVe vectors from here http://www-nlp.stanford.edu/data/glove.840B.300d.zip

# In[ ]:


get_ipython().system('wget  http://www-nlp.stanford.edu/data/glove.840B.300d.zip')


# In[ ]:


get_ipython().system('ls ../input/glove840b300dtxt/')


# In[ ]:


embeddings_index = {}
f = open('../input/glove840b300dtxt/glove.840B.300d.txt')
for line in tqdm.tqdm(f):
    values = line.split()
    word = ''.join(values[:-300])
    coefs = np.asarray(values[-300:], dtype='float32')
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
xtrain_glove = [sent2vec(x) for x in tqdm.tqdm(xtrain)]
xvalid_glove = [sent2vec(x) for x in tqdm.tqdm(xvalid)]


# In[ ]:


xtrain_glove = np.array(xtrain_glove)
xvalid_glove = np.array(xvalid_glove)


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


# ## Deep learning
# Let's first scale the data

# In[ ]:


# scale the data before any neural net:
scl = preprocessing.StandardScaler()
xtrain_glove_scl = scl.fit_transform(xtrain_glove)
xvalid_glove_scl = scl.transform(xvalid_glove)


# In[ ]:


# we need to binarize the labels for the neural net
ytrain_enc = np_utils.to_categorical(ytrain)
yvalid_enc = np_utils.to_categorical(yvalid)


# In[ ]:


# create a simple 3 layer sequential neural net
model = Sequential()

model.add(Dense(300, input_dim=300, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(300, activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(Dense(3))
model.add(Activation('softmax'))

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[ ]:


model.fit(xtrain_glove_scl, y=ytrain_enc, batch_size=64, 
          epochs=15, verbose=1, 
          validation_data=(xvalid_glove_scl, yvalid_enc))


# To move further, i.e. with LSTMs we need to tokenize the text data
# 

# In[ ]:


# using keras tokenizer here
token = text.Tokenizer(num_words=None)
max_len = 70

token.fit_on_texts(list(xtrain) + list(xvalid))
xtrain_seq = token.texts_to_sequences(xtrain)
xvalid_seq = token.texts_to_sequences(xvalid)

# zero pad the sequences
xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)

word_index = token.word_index


# In[ ]:


# create an embedding matrix for the words we have in the dataset
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm.tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# In[ ]:


from keras.layers.embeddings import Embedding

# A simple LSTM with glove embeddings and two dense layers
model = Sequential()
model.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
model.add(SpatialDropout1D(0.3))
model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[ ]:


model.fit(xtrain_pad, y=ytrain_enc, batch_size=512, epochs=100, verbose=1, validation_data=(xvalid_pad, yvalid_enc))


# In[ ]:


# A simple bidirectional LSTM with glove embeddings and two dense layers
model = Sequential()
model.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
model.add(SpatialDropout1D(0.3))
model.add(Bidirectional(LSTM(300, dropout=0.3, recurrent_dropout=0.3)))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Fit the model with early stopping callback
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
model.fit(xtrain_pad, y=ytrain_enc, batch_size=512, epochs=100, 
          verbose=1, validation_data=(xvalid_pad, yvalid_enc), callbacks=[earlystop])


# In[ ]:


# GRU with glove embeddings and two dense layers
model = Sequential()
model.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
model.add(SpatialDropout1D(0.3))
model.add(GRU(300, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
model.add(GRU(300, dropout=0.3, recurrent_dropout=0.3))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Fit the model with early stopping callback
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
model.fit(xtrain_pad, y=ytrain_enc, batch_size=512, epochs=100, 
          verbose=1, validation_data=(xvalid_pad, yvalid_enc), callbacks=[earlystop])


# Finally you can give a try with ensemble of all these models
