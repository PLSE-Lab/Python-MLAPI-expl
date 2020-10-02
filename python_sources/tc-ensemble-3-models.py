#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Increase batch size 
# Batch based prediction 
# Architecture - Batch Normalization, PReLU
# Add another sparse matrix formulation 
#-----------------
# Model 2 with Dropouts 


# In[22]:


import time 
import gc
start_time = time.time()

import numpy as np
np.random.seed(42)
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras import backend as K

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '4'

import threading
import multiprocessing
from multiprocessing import Pool, cpu_count
from contextlib import closing
cores = 4

#Get Training and test data 

train1 = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')

#train = train1["comment_text"].fillna("fillna").values
#test = test1["comment_text"].fillna("fillna").values

train = train1
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

max_features = 50000
maxlen = 100
embed_size = 300


# In[23]:


from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
 
lemmatizer = WordNetLemmatizer()
 
def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None
 
def clean_text(text):
    text = text.replace("<br />", " ")
    #text = text.decode("utf-8")
    return text

def swn_polarity(text):
    """
    Return a sentiment polarity: 0 = negative, 1 = positive
    """
    tokens_count = 0
    sentiment = 0.0
    text = clean_text(text)
    tagged_sentence = pos_tag(word_tokenize(text))
    for word, tag in tagged_sentence:
        wn_tag = penn_to_wn(tag)
        if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
            continue
        lemma = lemmatizer.lemmatize(word, pos=wn_tag)
        if not lemma:
            continue
        synsets = wn.synsets(lemma, pos=wn_tag)
        if not synsets:
            continue
            
        # Take the first sense, the most common
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())
        #print(word, lemma)
        #print(swn_synset.pos_score(), swn_synset.neg_score(), swn_synset.obj_score())
        
        if swn_synset.obj_score() == 1:
            sentiment += swn_synset.pos_score() - swn_synset.neg_score() 
        
        elif swn_synset.obj_score() != 1:
            sentiment += swn_synset.pos_score() - swn_synset.neg_score() + swn_synset.obj_score()
        
        tokens_count += 1
            
    return sentiment

# Function to be used in Parallelized data frame 
def sentiment(df):
    return df.apply(lambda x: swn_polarity(x))

# Parallelize data frame operation 
cores = 4
def parallelize_dataframe(df, func):
    df_split = np.array_split(df, cores)
    with closing(Pool(cores)) as pool:
        df = pd.concat(pool.map(func, df_split))
    return df

# Calculate Sentiment score 

def sentiment_score(df):
    df['senti_score'] = parallelize_dataframe(df['comment_text'], sentiment)
    return df


# In[24]:


# Calculate sentiment score for Training and test data 
import time 
t1 = time.time()
train = sentiment_score(train)
t2 = time.time()
print("Time taken is "+str(t2-t1))
print(train.shape)


# In[25]:


k1 = train['senti_score'].reshape(-1,1)


# In[26]:


from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
train['senti_score_scaled'] = scaler.fit_transform(k1)


# In[27]:


train['comment_text'] = train['comment_text'].astype(str)


# In[28]:


tokenizer = text.Tokenizer(num_words=max_features)
#all_text = np.hstack([test['comment_text'].str.lower(), train['comment_text'].str.lower()])
all_text = np.hstack([train['comment_text'].str.lower()])

tokenizer.fit_on_texts(all_text)

print("Fitting Done...Start text to sequence transform")

train['seq_comment']= tokenizer.texts_to_sequences(train.comment_text.str.lower())
print("Transform done for train ")


# In[29]:


#y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values


# In[30]:


x_train, x_valid, y_train, y_valid = train_test_split(train, y_train, train_size=0.95, random_state=233)
print(x_train.shape, x_valid.shape)
print(y_train.shape, y_valid.shape)


# In[31]:


def get_keras_data(dataset):
    X = { 
          'comment_text' : pad_sequences(dataset.seq_comment, maxlen=maxlen), 
          'senti_score_scaled': np.array(dataset.senti_score)
    } 
    return X


# In[32]:


X_train = get_keras_data(x_train)
X_valid = get_keras_data(x_valid)


# In[ ]:


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        print(self.interval)
        self.X_val, self.y_val = validation_data
        print(self.y_val)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))

def get_model1():
    
    inp = Input(shape = [X_train['comment_text'].shape[1]], name = 'comment_text')
    #senti_score = Input(shape=[1], name="senti_score_scaled")
    x = Embedding(max_features, embed_size)(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(40, return_sequences=True, dropout = 0.15, recurrent_dropout = 0.15))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(6, activation='sigmoid')(conc)
    
    model = Model([inp], outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

model1 = get_model1()

batch_size = 128
epochs = 1

RocAuc = RocAucEvaluation(validation_data=(X_valid, y_valid), interval=1)
os.environ['OMP_NUM_THREADS'] = '4'

exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(len(X_train)/batch_size) * epochs
lr_init, lr_fin = 0.009, 0.0045
lr_decay = exp_decay(lr_init, lr_fin, steps)
K.set_value(model1.optimizer.lr, lr_init)
K.set_value(model1.optimizer.decay, lr_decay)

for i in range(3):
    hist = model1.fit(X_train, y_train, batch_size=batch_size+(batch_size*(2*i)), epochs=epochs, validation_data=(X_valid,y_valid), callbacks=[RocAuc], verbose=1)

print("Model 1 Done")


# In[ ]:


import gensim 
w2v = gensim.models.KeyedVectors.load_word2vec_format('../input/googlenews-vectors-negative300/GoogleNews-vectors-negative300.bin', binary=True)
print("Done loading model")


# In[ ]:


vocab_size = len(tokenizer.word_index)+1
EMBEDDING_DIM = 300 # this is from the pretrained vectors
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
print(embedding_matrix.shape)
# Creating Embedding matrix 
c = 0 
c1 = 0 
w_Y = []
w_No = []
for word, i in tokenizer.word_index.items():
    if word in w2v:
        c +=1
        embedding_vector = w2v[word]
        w_Y.append(word)
    else:
        embedding_vector = None
        #embedding_vector = np.sum(embedding_matrix, axis = 0)
        w_No.append(word)
        c1 +=1
    if embedding_vector is not None:    
        embedding_matrix[i] = embedding_vector

print(c,c1, len(w_No), len(w_Y))
print(embedding_matrix.shape)

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        print(self.interval)
        self.X_val, self.y_val = validation_data
        print(self.y_val)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))
            
def get_model2():
    
    inp = Input(shape = [X_train['comment_text'].shape[1]], name = 'comment_text')
    #senti_score = Input(shape=[1], name="senti_score_scaled")
    #key_words_scaled = Input(shape=[1], name="key_words_scaled")
    print("Here")
    x = Embedding(vocab_size, EMBEDDING_DIM, weights = [embedding_matrix], trainable = True)(inp)

    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(40, return_sequences=True, dropout = 0.15, recurrent_dropout = 0.15))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(6, activation="sigmoid")(conc)
    
    model = Model([inp], outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

model2 = get_model2()

RocAuc = RocAucEvaluation(validation_data=(X_valid, y_valid), interval=1)
os.environ['OMP_NUM_THREADS'] = '4'

exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(len(X_train)/batch_size) * epochs
lr_init, lr_fin = 0.009, 0.0045
lr_decay = exp_decay(lr_init, lr_fin, steps)
K.set_value(model2.optimizer.lr, lr_init)
K.set_value(model2.optimizer.decay, lr_decay)

for i in range(3):
    hist = model2.fit(X_train, y_train, batch_size=batch_size+(batch_size*(2*i)), epochs=epochs, validation_data=(X_valid,y_valid), callbacks=[RocAuc], verbose=1)

print("Model 2 Done")


# In[ ]:


del X_train, X_valid, y_valid
gc.collect()


# In[ ]:


y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values


# # sklearn 
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.pipeline import Pipeline, FeatureUnion 
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.preprocessing import MinMaxScaler
# 
# class RocAucEvaluation(Callback):
#     def __init__(self, validation_data=(), interval=1):
#         super(Callback, self).__init__()
# 
#         self.interval = interval
#         print(self.interval)
#         self.X_val, self.y_val = validation_data
#         print(self.y_val)
# 
#     def on_epoch_end(self, epoch, logs={}):
#         if epoch % self.interval == 0:
#             y_pred = self.model.predict(self.X_val, verbose=0)
#             score = roc_auc_score(self.y_val, y_pred)
#             print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))
#             
# class TextSelector(BaseEstimator, TransformerMixin):
#     """
#     Transformer to select a single column from the data frame to perform additional transformations on
#     Use on text columns in the data
#     """
#     def __init__(self, key):
#         self.key = key
# 
#     def fit(self, X, y=None):
#         return self
# 
#     def transform(self, X):
#         return X[self.key]
#     
# class NumberSelector(BaseEstimator, TransformerMixin):
#     """
#     Transformer to select a single column from the data frame to perform additional transformations on
#     Use on numeric columns in the data
#     """
#     def __init__(self, key):
#         self.key = key
# 
#     def fit(self, X, y=None):
#         return self
# 
#     def transform(self, X):
#         return X[[self.key]]
#     
# from sklearn.pipeline import make_pipeline
# from sklearn.linear_model import LogisticRegressionCV
# 
# p1 = Pipeline([('selector', TextSelector(key='comment_text')),('comment_text', TfidfVectorizer( stop_words='english', max_features = 50000, ngram_range =(1,2)))])
# p2 = Pipeline([('selector', NumberSelector(key='senti_score_scaled')),('senti_score_scaled', MinMaxScaler() )])
# vectorizer = FeatureUnion([('comment_text', p1), ('senti_score_scaled', p2)], n_jobs =4)
# 
# import time
# t1 = time.time()
# vectorizer.fit(train)
# print("Vectorizer Fitting done")
# X_train = vectorizer.transform(train).astype(np.float32)
# print(type(X_train))
# print("Vectorizer Transformation completed")
# 
# print(X_train.shape, y_train.shape)
# X_train1, X_valid1, y_train1, y_valid1 = train_test_split(X_train, y_train, train_size=0.95, random_state=233)
# print(X_train1.shape, X_valid1.shape)
# print(y_train1.shape, y_valid1.shape)
# 
# def get_model5():
#     
#     inp = Input(shape = [X_train.shape[1]], sparse = True)
#     #senti_score = Input(shape=[1], name="senti_score_scaled")
#     #key_words_scaled = Input(shape=[1], name="key_words_scaled")
#     #x = Embedding(max_features, embed_size)(inp)
#     #x = Embedding(vocab_size, EMBEDDING_DIM, weights = [embedding_matrix], trainable = True)(inp)
#     #x = concatenate([x, senti_score])
#     #x = BatchNormalization()(x)
#     
#     x  = Dense(128, activation="sigmoid")(inp)
#     x  = Dense(64, activation="sigmoid")(x)
#     
#     #conc = concatenate([x])
# 
#     outp = Dense(6, activation="sigmoid")(x)
#     
#     model = Model([inp], outp)
#     model.compile(loss='binary_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])
# 
#     return model
# 
# model5 = get_model5()
# 
# RocAuc = RocAucEvaluation(validation_data=(X_valid1, y_valid1), interval=1)
# os.environ['OMP_NUM_THREADS'] = '4'
# 
# batch_size = 64
# epochs = 1
# 
# exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
# steps = int(X_train1.shape[0]/batch_size) * epochs
# lr_init, lr_fin = 0.001, 0.0005
# lr_decay = exp_decay(lr_init, lr_fin, steps)
# K.set_value(model5.optimizer.lr, lr_init)
# K.set_value(model5.optimizer.decay, lr_decay)
# 
# for i in range(5):
#     hist = model5.fit(x = X_train1, y = y_train1, batch_size=batch_size+(batch_size*(2*i)), epochs=epochs, validation_data=(X_valid1,y_valid1), callbacks=[RocAuc], verbose=1)
# 
# print("Model 5 Done")

# In[ ]:


t1 = time.time()

import gc
def load_test():
    for df in pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv', chunksize= 150000):
        yield df

print("Yield complete")

test_ids = np.array([], dtype=np.int32)
preds= np.zeros((0,6), dtype = np.int32)

print("Start Batch Prediction")
c = 0 

for df in load_test():
    
    c +=1
    print("Chunk number is "+str(c))
    sentiment_score(df)
    k2 = df['senti_score'].reshape(-1,1)
    df['senti_score_scaled'] = scaler.transform(k2)
    df['seq_comment'] = tokenizer.texts_to_sequences(df.comment_text.str.lower())
    print("Transform done for test ")
    X_test = get_keras_data(df)
    y_pred1 = model1.predict(X_test, batch_size=1024, verbose =1)
    y_pred2 = model2.predict(X_test, batch_size=1024, verbose =1)
    
    test_id = df['id']
    del df['seq_comment'], df['senti_score'], X_test
    gc.collect()
    
    #X_test = vectorizer.transform(df).astype(np.float32)
    #y_pred5 = model5.predict(X_test, batch_size=1024, verbose =1)
    #print(y_pred5.shape)
    
    k = (y_pred1+y_pred2)/2
    print(k.shape)
    preds= np.append(preds,k, axis = 0)
    print(preds.shape)
    test_ids = np.append(test_ids, test_id)

print("All chunks done")
t2 = time.time()
print("Total time for Parallel Batch Prediction is "+str(t2-t1))


# In[ ]:


submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')
#y_pred1 = model.predict(X_test, batch_size=1024, verbose =1)
#y_pred2 = model1.predict(X_test, batch_size=1024, verbose =1)
#y_pred3 = model3.predict(X_test, batch_size=1024, verbose =1)
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = preds
submission.to_csv('submission.csv', index=False)
end_time = time.time()
print("Total time taken is "+str(end_time-start_time))


# In[ ]:


y_pred1[0]


# In[ ]:


y_pred2[0]

