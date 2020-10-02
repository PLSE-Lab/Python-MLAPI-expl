#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing import text, sequence
from keras.layers import GRU, Embedding, Dense, Input
from keras.models import Model
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv")
test = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/test.csv")


# In[ ]:


# Baseline Keras model
# 1) padded sequences 
# 2) FastText Web Crawl embedding 
# 3) single GRU layer. 

max_features = 10000
maxlen = 50

X_train = train.comment_text.fillna('na').values
X_test = test.comment_text.fillna('na').values
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train_sequence = tokenizer.texts_to_sequences(X_train)
X_test_sequence = tokenizer.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train_sequence, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test_sequence, maxlen=maxlen)
print(len(tokenizer.word_index))


# In[ ]:


# Load the FastText Web Crawl vectors
EMBEDDING_FILE_FASTTEXT="../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec"
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index_ft = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE_FASTTEXT,encoding='utf-8'))


# In[ ]:


embed_size=300
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index_ft.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    


# In[ ]:


# Baseline Model:  padded sequences (max_features: 10,000, maxlen: 50, FastText Web Crawl, Embed Size=300, 1 GRU layer, num_filters=40)
# Result:(Private: 9769, Public: 9775)
# Uncomment to run baseline.
def get_model(maxlen, max_features, embed_size, num_filters=40):
    inp = Input(shape=(maxlen, ))

    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    
    x = GRU(num_filters, )(x)
    
    outp = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=outp)
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


# In[ ]:


# Avg Pooling Model:  padded sequences 
# Result:(Private: 9769, Public: 9775)
# Uncomment to run baseline.
from keras.layers import GlobalAveragePooling1D

def get_model2(maxlen, max_features, embed_size, num_filters=40):
    inp = Input(shape=(maxlen, ))

    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    
    x = GRU(num_filters, return_sequences=True)(x)
    
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    
    outp = Dense(6, activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=outp)
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


# In[ ]:





model = get_model2(maxlen, max_features, embed_size)

batch_size = 32
epochs = 2

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.95, random_state=233)

model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), verbose=1)

y_pred = model.predict(x_test, batch_size=batch_size,verbose=1)

sample_submission = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv")
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
sample_submission[class_names] = y_pred

sample_submission.to_csv('pooled_submission.csv',index=False)


# In[ ]:


# Improvement #1: Preprocessing.  Clean up data before padding and vectorizing.
#
# I am using the output from xbf: Processing helps boosting about 0.0005 on LB

'''
train = pd.read_csv("../input/processing-helps-boosting-about-0-0005-on-lb/train_processed.csv")
test = pd.read_csv("../input/processing-helps-boosting-about-0-0005-on-lb/test_processed.csv")

X_train = train.comment_text.fillna('na').values
X_test = test.comment_text.fillna('na').values
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train_sequence = tokenizer.texts_to_sequences(X_train)
X_test_sequence = tokenizer.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train_sequence, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test_sequence, maxlen=maxlen)
print(len(tokenizer.word_index))

embed_size=300
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index_ft.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        
model = get_model(maxlen, max_features, embed_size)

batch_size = 32
epochs = 

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.95, random_state=233)

model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), verbose=1)

y_pred = model.predict(x_test, batch_size=batch_size,verbose=1)

sample_submission = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv")
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
sample_submission[class_names] = y_pred

sample_submission.to_csv('preprocess_submission.csv',index=False)
'''


# In[ ]:


# Improvement #2:  Run until roc auc score stops increasing using a callback
# result:  (Private: 9766, Public: 9783)

from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score

'''
class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.max_score = 0
        self.not_lower_count = 0

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=1)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))
            if (score > self.max_score):
                print("*** New High Score (previous: %.6f) \n" % self.max_score)
                model.save_weights("best_weights.h5")
                self.max_score=score
                self.not_lower_count = 0
            else:
                self.not_lower_count += 1
                # in my code, I use 3 instead of 2.
                if self.not_lower_count > 2:
                    print("Epoch %05d: early stopping, high score = %.6f" % (epoch,self.max_score))
                    self.model.stop_training = True

model = get_model(maxlen, max_features, embed_size)

batch_size = 32
epochs = 100

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.95, random_state=233)

RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)

model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks=[RocAuc], verbose=1)

model.load_weights("best_weights.h5")

y_pred = model.predict(x_test, batch_size=batch_size,verbose=1)

sample_submission = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv")
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
sample_submission[class_names] = y_pred

sample_submission.to_csv('improvement1_submission.csv',index=False,)
'''


# In[ ]:




