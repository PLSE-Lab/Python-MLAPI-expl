#!/usr/bin/env python
# coding: utf-8

# # Comparison of different RNN architectures
# In this notebook, we will compare different architectures of Recurrent Neural Networks. We will use data from the competition [Jigsaw Multilingual Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification). We will use jigsaw-toxic-comment-train.csv file and divide it into train and test datasets.<br><br>
# We will use 50 dimensional [Glove vectors](https://nlp.stanford.edu/projects/glove/) as word embeddings.
# 
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
from tqdm.notebook import tqdm
from keras.preprocessing.text import Tokenizer
from keras.layers import  Embedding, Input, LSTM, Dense, SimpleRNN, Bidirectional
from keras.initializers import Constant
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.metrics import AUC
from sklearn.metrics import accuracy_score, roc_auc_score

random.seed(0)


# In[ ]:


## Hyperparameters

TEST_PROPORTION = 0.2 #proportion of dataset to be used for testing
MAX_VOCAB_SIZE = 100000 #Maximum vocabulary size
MAX_SEQ_LEN = 128 #Maximum sequence length for comments
EMBEDDING_SIZE = 50 #Size of word embedding vector


# In[ ]:


## Reading Data

data = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')
print ("Number of rows in data: ", data.shape[0])


# In[ ]:


data.head()


# We will only use two columns -
# 1. comment_text 
# 2. toxic - label

# In[ ]:


## Function to split data into train and test sets 

def train_test_split(df,test_prop=0.25):
    n_rows = df.shape[0]
    list_indices = list(range(n_rows))
    random.shuffle(list_indices)
    n_rows_test = int(n_rows*test_prop)
    n_rows_train = n_rows - n_rows_test
    df_train = df.iloc[list_indices[:n_rows_train]]
    df_test = df.iloc[list_indices[n_rows_train:]]
    return df_train, df_test
    


# In[ ]:


## Function to read Glove Embeddings

def read_glove_vecs(glove_file):
    words = []
    word_to_vec_map = {}
    with open(glove_file,'r') as f:
        for line in f:
            line = line.strip().split()
            words += [line[0]]
            word_to_vec_map[line[0]] = np.array(line[1:], dtype = np.float64)
    return words, word_to_vec_map


# In[ ]:


data_train,data_test =train_test_split(data, TEST_PROPORTION)
print ("Number of rows in train dataset: ", data_train.shape)
print ("Number of rows in test dataset: ", data_test.shape)


# In[ ]:


## Reading the Glove embeddings

words, word_to_vec_map = read_glove_vecs('../input/glove6b50dtxt/glove.6B.50d.txt')
print ("Size of vocabulary: ", len(words))
print ("Length of word vector: ", len(word_to_vec_map['bat']))


# In[ ]:


## creating tokenizer

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(data_train.comment_text)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n## tokenizing training and testing data \n\nsequences_train = tokenizer.texts_to_sequences(data_train.comment_text)\nsequences_test = tokenizer.texts_to_sequences(data_test.comment_text)')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n## creating the training data and testing data \n\n## padding the sequences to meet the maximum sequence length\ntrain_x = pad_sequences(sequences_train, maxlen=MAX_SEQ_LEN)\ntest_x = pad_sequences(sequences_test, maxlen=MAX_SEQ_LEN)\n\ntrain_y = data_train.toxic\ntest_y = data_test.toxic\n\nprint ("Shape of training data features: ",train_x.shape)\nprint ("Shape of training data labels: ",train_y.shape)\nprint ("Shape of testing data features: ",test_x.shape)\nprint ("Shape of testing data labels: ",test_y.shape)')


# In[ ]:


## Creating word embeddings matrix from glove embeddings

# Initializing the matrix with zeros
embedding_matrix = np.zeros((MAX_VOCAB_SIZE, EMBEDDING_SIZE))

for word, i in tqdm(tokenizer.word_index.items()):
    if i >= EMBEDDING_SIZE:
        continue
    vec = word_to_vec_map.get(word)
    if vec is not None:
        embedding_matrix[i,:]=vec


# In[ ]:


## Creating embedding layer

embedding_layer = Embedding(MAX_VOCAB_SIZE,
                            EMBEDDING_SIZE,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQ_LEN,
                            trainable=False)


# In[ ]:


## Defining different architectures of RNN

# RNN with simple unit
def model_SimpleRNN(input_shape):
    input_word_ids = Input(shape=(input_shape,), dtype='int32', name="input_word_ids")
    embedded_sequences = embedding_layer(input_word_ids)
    lstm_output = SimpleRNN(128,return_sequences=False)(embedded_sequences)
    out = Dense(1, activation='sigmoid')(lstm_output)
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=[AUC()])
    return model

# RNN with LSTM unit
def model_LSTM(input_shape):
    input_word_ids = Input(shape=(input_shape,), dtype='int32', name="input_word_ids")
    embedded_sequences = embedding_layer(input_word_ids)
    lstm_output = LSTM(128,return_sequences=False)(embedded_sequences)
    out = Dense(1, activation='sigmoid')(lstm_output)
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=[AUC()])
    return model


# RNN with 2 layers of LSTM units
def model_LSTM_deep(input_shape):
    input_word_ids = Input(shape=(input_shape,), dtype='int32', name="input_word_ids")
    embedded_sequences = embedding_layer(input_word_ids)
    lstm_output = LSTM(128,return_sequences=True)(embedded_sequences)
    lstm_output2 = LSTM(128,return_sequences=False)(lstm_output)
    out = Dense(1, activation='sigmoid')(lstm_output2)
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=[AUC()])
    return model

# RNN with 2 layers of bi-directional LSTM units
def model_LSTM_deep_bi(input_shape):
    input_word_ids = Input(shape=(input_shape,), dtype='int32', name="input_word_ids")
    embedded_sequences = embedding_layer(input_word_ids)
    lstm_output = Bidirectional(LSTM(128,return_sequences=True))(embedded_sequences)
    lstm_output2 = LSTM(128,return_sequences=False)(lstm_output)
    out = Dense(1, activation='sigmoid')(lstm_output2)
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=[AUC()])
    return model


# ## Simple RNN

# In[ ]:


# Initializing the model

model_simple = model_SimpleRNN(MAX_SEQ_LEN)
model_simple.summary()


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Training the model\n\nmodel_simple.fit(train_x, train_y,\n          batch_size=128,\n          epochs=1)')


# In[ ]:


predictions_simple = model_simple.predict(test_x, batch_size=128)
auc_simple = roc_auc_score(test_y, predictions_simple)
print ("AUC for simple RNN unit: ",auc_simple)


# ## LSTM

# In[ ]:


# Initializing the model

model_lstm = model_LSTM(MAX_SEQ_LEN)
model_lstm.summary()


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Training the model\n\nmodel_lstm.fit(train_x, train_y,\n          batch_size=128,\n          epochs=1)')


# In[ ]:


predictions_lstm = model_lstm.predict(test_x, batch_size=128)
auc_lstm = roc_auc_score(test_y, predictions_lstm)
print ("AUC for LSTM unit: ",auc_lstm)


# ## 2 Layers of LSTM

# In[ ]:


# Initializing the model

model_lstm_deep = model_LSTM_deep(MAX_SEQ_LEN)
model_lstm_deep.summary()


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Training the model\n\nmodel_lstm_deep.fit(train_x, train_y,\n          batch_size=128,\n          epochs=1)')


# In[ ]:


predictions_lstm_deep = model_lstm_deep.predict(test_x, batch_size=128)
auc_lstm_deep = roc_auc_score(test_y, predictions_lstm_deep)
print ("AUC for 2 Layers LSTM units: ",auc_lstm_deep)


# ## 2 Layers of Bi-directional LSTM

# In[ ]:


# Initializing the model

model_lstm_deep_bi = model_LSTM_deep_bi(MAX_SEQ_LEN)
model_lstm_deep_bi.summary()


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Training the model\n\nmodel_lstm_deep_bi.fit(train_x, train_y,\n          batch_size=128,\n          epochs=1)')


# In[ ]:


predictions_lstm_deep_bi = model_lstm_deep_bi.predict(test_x, batch_size=128)
auc_lstm_deep_bi = roc_auc_score(test_y, predictions_lstm_deep_bi)
print ("AUC for 2 Layers Bi-directional LSTM units: ",auc_lstm_deep_bi)


# In[ ]:




