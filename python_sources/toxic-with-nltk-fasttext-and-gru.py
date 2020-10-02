#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
import nltk
from nltk.corpus import stopwords
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
os.environ['OMP_NUM_THREADS'] = '4'
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[3]:


stops = set(stopwords.words('english'))


# In[5]:


def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df


# In[29]:


def removing_stopwords(df, text_field):
    for i in range(len(df[text_field])):
        df[text_field][i] = ' '.join([word for word in df[text_field][i].split() if word not in stops])
    return df


# In[25]:


EMBEDDING_FILE = '../input/fast-text-vector/crawl-300d-2M.vec'


# In[35]:


train = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv")
test = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/test.csv")


# In[37]:


train["comment_text"].fillna("fillna")
test["comment_text"].fillna("fillna")
train = removing_stopwords(train,"comment_text")
test = removing_stopwords(test,"comment_text")


# In[39]:


train = standardize_text(train,"comment_text")
test = standardize_text(test,"comment_text")


# In[40]:


train_x = train["comment_text"].values
train_y = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
test_x = test["comment_text"].values


# In[41]:


max_features = 30000
maxlen = 100
embed_size = 300


# In[42]:


tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_x) + list(test_x))


# In[43]:


train_x = tokenizer.texts_to_sequences(train_x)
test_x = tokenizer.texts_to_sequences(test_x)


# In[44]:


train_x = sequence.pad_sequences(train_x, maxlen=maxlen)
test_x = sequence.pad_sequences(test_x, maxlen=maxlen)


# In[ ]:


def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')


# In[ ]:


embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))


# In[ ]:


word_index = tokenizer.word_index


# In[ ]:


nb_words = min(max_features, len(word_index))


# In[ ]:


embedding_matrix = np.zeros((nb_words, embed_size))


# In[ ]:


for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


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


def get_model():
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(6, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

model = get_model()


# In[ ]:


model.summary()


# In[ ]:


batch_size = 32
epochs = 2


# In[ ]:


X_tra, X_val, y_tra, y_val = train_test_split(train_x, train_y, train_size=0.95, random_state=233)


# In[ ]:


RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)


# In[ ]:


hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                 callbacks=[RocAuc], verbose=2)


# In[ ]:


y_pred = model.predict(test_x, batch_size=1024)


# In[ ]:


submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')


# In[ ]:


submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('submission.csv', index=False)


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




