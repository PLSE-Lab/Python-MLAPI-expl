#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[36]:


import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalAveragePooling1D,GlobalMaxPooling1D
from keras.models import Model
from keras.preprocessing import text, sequence
from keras import initializers, regularizers, constraints, optimizers, layers

from keras.layers import Conv1D, MaxPooling1D, Activation, GRU
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D,concatenate
from keras.callbacks import Callback
import warnings
warnings.filterwarnings('ignore')

from keras.models import Sequential

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# Code is an adaptation of Jeremy Howard( https://www.kaggle.com/jhoward/improved-lstm-baseline-glove-dropout-lb-0-048) code

# In[18]:


path = '../input/'
comp = 'jigsaw-toxic-comment-classification-challenge/'
EMBEDDING_FILE = '../input/glove840b300dtxt/glove.840B.300d.txt'
##EMBEDDING_FILE= '../input/glove-global-vectors-for-word-representation/glove.6B.50d.txt'
TRAIN_DATA_FILE=f'{path}{comp}train.csv'
TEST_DATA_FILE=f'{path}{comp}test.csv'


# In[19]:


train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)
submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')
X_train = train["comment_text"].fillna("fillna").values
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_test = test["comment_text"].fillna("fillna").values


# In[20]:


embed_size = 300 # how big is each word vector
max_features = 100000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 200 


# In[21]:


tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)


# In[22]:


def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))


# In[23]:


word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# In[24]:


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


# ## Convolutional Network for NLP

# In[29]:


def get_model():
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features,embed_size,weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.3)(x)
    
    conv_0 = Conv1D(64, kernel_size=3, kernel_initializer='normal',activation='elu',padding='same')(x)
    maxpool_0 = MaxPooling1D()(conv_0)
    conv_1 = Conv1D(64, kernel_size=5, kernel_initializer='normal',activation='elu',padding='same')(x)
    maxpool_1 = MaxPooling1D()(conv_1)
    conv_2 = Conv1D(64, kernel_size=6, kernel_initializer='normal',activation='elu',padding='same')(x)
    maxpool_2 = MaxPooling1D()(conv_2)
    z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2]) 
    z = Bidirectional(GRU(40, return_sequences=True))(z)
    avg_pool = GlobalAveragePooling1D()(z)
    max_pool = GlobalMaxPooling1D()(z)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(6, activation="sigmoid")(conc)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


# In[ ]:


#model = Sequential ([
#    Embedding(max_features,embed_size , input_length=maxlen, weights=[embedding_matrix], trainable=True),
#    Dropout(0.2),
#    Conv1D(32, kernel_size=3, padding='same', activation="relu"),
#    MaxPooling1D(),
#    Conv1D(64, kernel_size=4, padding='same', activation="relu"),
#    MaxPooling1D(),
#    Conv1D(128, kernel_size=5, padding='same', activation="relu"),
#    MaxPooling1D(),
#    Dropout (0.5),
#    GRU(50,return_sequences=True),
#    Dropout (0.25),
#    Flatten(),
#    Dense (100, activation="relu"),
#    Dropout (0.45),
#    Dense (6, activation='sigmoid')
#    ])


# In[ ]:


#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


#model.summary()


# In[ ]:


#model.fit(X_t, y, batch_size=64, epochs=5,validation_split=0.1)


# In[28]:


from keras import backend as K
import tensorflow as tf
session_conf = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)
K.set_session(tf.Session(graph=tf.get_default_graph(), config=session_conf))


# In[34]:


model = get_model()


# In[37]:


batch_size = 32
epochs = 2

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.95, random_state=233)

RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)


# In[ ]:


hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=2, validation_data=(X_val, y_val),
                 callbacks=[RocAuc], verbose=2)


# In[ ]:


y_pred = model.predict(x_test, batch_size=1024)
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('submission.csv', index=False)


# # Submit the data

# In[ ]:


#y_test = model.predict([X_te], batch_size=1024, verbose=1)
#sample_submission = pd.read_csv(f'{path}{comp}sample_submission.csv')
#sample_submission[list_classes] = y_test
#sample_submission.to_csv('submission.csv', index=False)


# In[ ]:




