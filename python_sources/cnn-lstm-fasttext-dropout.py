#!/usr/bin/env python
# coding: utf-8

# # Improved LSTM baseline
# 
# This kernel is a somewhat improved version of [Keras - Bidirectional LSTM baseline](https://www.kaggle.com/CVxTz/keras-bidirectional-lstm-baseline-lb-0-051) 

# In[ ]:


import sys, os, re, csv, codecs, numpy as np, pandas as pd
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

from keras.callbacks import Callback
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, concatenate, MaxPooling1D
from keras.layers import Bidirectional, GlobalMaxPooling1D, Flatten, TimeDistributed, Conv1D, SpatialDropout1D, GlobalAveragePooling1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers


# We include the GloVe word vectors in our input files. 

# In[ ]:


path = '../input/'
comp = 'jigsaw-toxic-comment-classification-challenge/'
EMBEDDING_FILE=f'{path}glove840b300dtxt/glove.840B.300d.txt'
TRAIN_DATA_FILE=f'{path}{comp}train.csv'
TEST_DATA_FILE=f'{path}{comp}test.csv'


# Set some basic config parameters:

# In[ ]:


embed_size = 300 # how big is each word vector
max_features = 150000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 150  # max number of words in a comment to use


# Read in our data and replace missing values:

# In[ ]:


train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)

list_sentences_train = train["comment_text"].fillna("_na_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("_na_").values
#print(list_sentences_train[:3])


# Standard keras preprocessing, to turn each comment into a list of word indexes of equal length (with truncation or padding as needed).

# In[ ]:


tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
#print(list_tokenized_train[:3])


# Read the glove word vectors (space delimited strings) into a dictionary from word->vector.

# In[ ]:


#def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
#embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))
embeddings_index = {}
with open(EMBEDDING_FILE,encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs


# Use these vectors to create our embedding matrix, with random initialization for words that aren't in GloVe. We'll use the same mean and stdev of embeddings the GloVe has when generating the random init.

# In[ ]:


word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# Simple bidirectional LSTM with two fully connected layers. We add some dropout to the LSTM since even 2 epochs is enough to overfit.

# In[ ]:


inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
#x = SpatialDropout1D(0.2)(x)
x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "glorot_uniform")(x)
x = Bidirectional(LSTM(100, return_sequences=True,dropout=0.05,recurrent_dropout=0.05))(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
x = concatenate([avg_pool, max_pool])

x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
adam = optimizers.Adam(lr=0.0015)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])


# In[ ]:


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X_t, y, test_size = 0.05, random_state = 144)
RocAuc = RocAucEvaluation(validation_data=(x_test, y_test), interval=1)


# Now we're ready to fit out model! Use `validation_split` when not submitting.

# In[ ]:


model.fit(x_train, y_train, batch_size=64, epochs=2, validation_data=(x_test, y_test), callbacks=[RocAuc]);


# And finally, get predictions for the test set and prepare a submission CSV:

# In[ ]:


y_test = model.predict([X_te], batch_size=1024, verbose=1)
sample_submission = pd.read_csv(f'{path}{comp}sample_submission.csv')
sample_submission[list_classes] = y_test
sample_submission.to_csv('submission.csv', index=False)

