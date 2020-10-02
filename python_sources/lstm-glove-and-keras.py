#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nltk import word_tokenize
from imblearn.under_sampling import RandomUnderSampler

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ###### 1. Read data
# Read the data from CSV and apply some basic pre-processing (remove non-ascii characters, convert our target variable to an integer label).

# In[ ]:


train_df = pd.read_csv(r'../input/train.csv')


# In[ ]:


train_df.head()


# In[ ]:


train_df.shape


# In[ ]:


docs = train_df["question_text"].values
labels = train_df["target"].values


# In[ ]:


np.unique(labels,return_counts=True)


# ### 2. Preprocessing
# Tokenize text, convert words / tokens to indexed integers. Take each document and convert to a sequence of max length 20 (pad with zeroes if shorter).

# In[ ]:


# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(docs)
print(vocab_size)


# In[ ]:


# pad documents to a max length of 20 words
max_length = 20
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(len(padded_docs))


# In[ ]:


padded_docs.shape


# In[ ]:


rus = RandomUnderSampler(random_state=42)
padded_docs_rus,labels_rus = rus.fit_resample(padded_docs,labels)


# In[ ]:


padded_docs_rus.shape,labels_rus.shape,padded_docs.shape,labels.shape


# In[ ]:


np.unique(labels_rus,return_counts=True)


# ### 3. Import embeddings
# The clever part: import a dictionary of word embeddings that translates each word into a 300 dimensional vector.

# In[ ]:


# load the whole embedding into memory
EMBEDDING_FILE = r"../input/embeddings/glove.840B.300d/glove.840B.300d.txt"
EMBEDDING_DIM = 100

def get_embedding():
    embeddings_index = {}
    f = open(EMBEDDING_FILE,'r', errors = 'ignore', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
        except:
            pass
            
    f.close()
    return embeddings_index

embeddings_index = get_embedding()


# In[ ]:


# create a weight matrix for words in training docs

def create_embedding_weights(vocab_size,t):
    embedding_matrix = np.zeros((vocab_size, 300))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print(embedding_matrix.shape)
    return embedding_matrix


# In[ ]:


embedding_matrix = create_embedding_weights(vocab_size,t)


# ### 4. Network architecture

# In[ ]:


## create model
model_glove = Sequential()
model_glove.add(Embedding(vocab_size, 300, input_length=20, weights=[embedding_matrix], trainable=False))
model_glove.add(Dropout(0.2))
model_glove.add(Conv1D(64, 5, activation='relu'))
model_glove.add(MaxPooling1D(pool_size=4))
model_glove.add(LSTM(100))
model_glove.add(Dense(1, activation='sigmoid'))
model_glove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


model_glove.summary()


# ### 5. Training and Evaluation
# Is it any good? Let's find out.
# Divide our dataset using a holdout strategy:

# In[ ]:


# split dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(padded_docs_rus, labels_rus, test_size=0.1, random_state=42)


# In[ ]:


filepath="/weights-{epoch:02d}-{val_acc:.4f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_best_only=False, save_weights_only=True, 
                             mode='auto', period=10)


# In[ ]:


earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5,
                          verbose=1, mode='auto')


# In[ ]:


# fit the model
model_glove.fit(X_train, y_train, epochs=50,validation_data=(X_test, y_test),
          verbose=1,callbacks=[checkpoint,earlystop])


# In[ ]:


# evaluate the model
loss, accuracy = model_glove.evaluate(X_test, y_test, verbose=1)
print('Accuracy: %f' % (accuracy*100))


# In[ ]:


model_glove.predict_classes(X_test)


# In[ ]:


ques = ['Has the United States become the largest dictatorship in the world?','Why do so many women become so rude and arrogant when they get just a little bit of wealth and power?','How should I prepare for IIT K/IIM C/ ISI K PGDBA course exam and interview?']


# In[ ]:


t.fit_on_texts(ques)
# integer encode the documents
encoded_ques = t.texts_to_sequences(ques)
max_length = 20
padded_ques = pad_sequences(encoded_ques, maxlen=max_length, padding='post')
print(len(padded_ques))


# In[ ]:


model_glove.predict_classes(padded_ques)


# In[ ]:


test_file = r"../input/test.csv"
test_df = pd.read_csv(test_file)


# In[ ]:


test_df.head()


# In[ ]:


test_docs = test_df['question_text'].values


# In[ ]:


t.fit_on_texts(test_docs)
# integer encode the documents
encoded_ques_test = t.texts_to_sequences(test_docs)
vocab_size_test = len(t.word_index) + 1
print(vocab_size_test)
max_length = 20
padded_ques_test = pad_sequences(encoded_ques_test, maxlen=max_length, padding='post')
print(len(padded_ques_test))


# In[ ]:


predicted_output = model_glove.predict_classes(padded_ques_test)


# In[ ]:


test_df['predicted_labels'] = predicted_output


# In[ ]:


test_df[test_df['predicted_labels'] == 1].head()


# In[ ]:


test_df.to_csv(r'submission.csv',index=False)


# In[ ]:




