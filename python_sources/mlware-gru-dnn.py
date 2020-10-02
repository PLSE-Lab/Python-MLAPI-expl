#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import gc
import tensorflow as tf
import datetime

from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Bidirectional, Conv1D, GRU, GlobalMaxPooling1D, GlobalAveragePooling1D, Embedding, concatenate, Dropout, BatchNormalization, Flatten, LSTM
from keras.preprocessing import text, sequence
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline


# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))


# In[ ]:


data = pd.read_csv("../input/technex/train.csv")
data.info()


# In[ ]:


# data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
data['domain'].fillna(value='', inplace=True)
data['title'].fillna(value='', inplace=True)
data['over_18'].fillna(value=0, inplace=True)
data['time_submitted'].fillna(value=0, inplace=True)
data['submitter_id'].fillna(value = 0, inplace = True)
data['title_len'] = len(data['title'].str.split())
data['hour'] = pd.to_datetime(data.time_submitted).dt.hour
data.info()


# In[ ]:


l1 = []
max_words = 0

for item in data['title']:
    l1 = l1 + [len(item)]
    max_words += len(item.split())
    
l2 = []

for item in data['domain']:
    l2 = l2 + [len(item)]
    max_words += len(item.split())
    

l1.sort()
l2.sort()

print(l1[-1])
print(l2[-1])
print(max_words)


# In[ ]:


def f1_score(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


# In[ ]:


test = pd.read_csv('../input/technex/test.csv')

test['domain'].fillna(value='', inplace=True)
test['title'].fillna(value='', inplace=True)
test['over_18'].fillna(value=0, inplace=True)
test['time_submitted'].fillna(value=0, inplace=True)
test['submitter_id'].fillna(value = 0, inplace = True)
test['title_len'] = len(test['title'].str.split())
test['hour'] = pd.to_datetime(test.time_submitted).dt.hour


# In[ ]:


data['title_domain']= (data['title'] + " " + data['domain']).astype(str)   
test['title_domain'] = (test['title'] + " " + test['domain']).astype(str)
data['title'] = data['title'].astype(str)
data['domain'] = data['domain'].astype(str)
test['title'] = test['title'].astype(str)
test['domain'] = test['domain'].astype(str)

max_seq_length = 100
max_words = 500000

tokenizer = text.Tokenizer(num_words = max_words)
all_text = np.hstack([data['title_domain'].str.lower(), test['title_domain'].str.lower()])
tokenizer.fit_on_texts(all_text)
del all_text
gc.collect()

data['seq_title'] = tokenizer.texts_to_sequences(data.title.str.lower())
data['seq_domain'] = tokenizer.texts_to_sequences(data.domain.str.lower())
del data['title']
del data['domain']
del data['title_domain']
gc.collect()

test['seq_title'] = tokenizer.texts_to_sequences(test.title.str.lower())
test['seq_domain'] = tokenizer.texts_to_sequences(test.domain.str.lower())
del test['title_domain']
del test['title']
del test['domain']
gc.collect()

data.info()


# In[ ]:


EMBEDDING_DIM1 = 300
EMBEDDING_FILE1 = '../input/fasttext/crawl-300d-2M.vec'
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index1 = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE1))


vocab_size = len(tokenizer.word_index)+2
EMBEDDING_DIM1 = 300
embedding_matrix1 = np.zeros((vocab_size, EMBEDDING_DIM1))
print(embedding_matrix1.shape) 
c = 0 
c1 = 0 
w_Y = []
w_No = []
for word, i in tokenizer.word_index.items():
    if word in embeddings_index1:
        c +=1
        embedding_vector = embeddings_index1[word]
        w_Y.append(word)
    else:
        embedding_vector = None
        w_No.append(word)
        c1 +=1
    if embedding_vector is not None:    
        embedding_matrix1[i] = embedding_vector

print(c,c1, len(w_No), len(w_Y))
print(embedding_matrix1.shape)
del embeddings_index1
gc.collect()

print(" FAST TEXT DONE")

print(vocab_size)


# In[ ]:


data.info()


# In[ ]:


Y = data['engaging']
X = data.copy(deep = True)

del X['engaging']
gc.collect()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0)


# In[ ]:


def get_dict(DF):

    data_dict = {}

    columns = list(DF.columns.values)
    
    text_col = ['seq_title', 'seq_domain']
    
    for column in columns:
        if(column in text_col):
            data_dict.update({column: pad_sequences(DF[column], maxlen = max_seq_length)})
        else:
            data_dict.update({column: np.array(DF[column])})
    
    return data_dict


# In[ ]:


#Inputs
seq_title = Input(shape=[100], name="seq_title")
seq_domain = Input(shape=[100], name="seq_domain")


over_18 = Input(shape=[1], name="over_18")
time_submitted = Input(shape=[1], name="time_submitted")
submitter_id = Input(shape=[1], name="submitter_id")
title_len = Input(shape=[1], name="title_len")
hour = Input(shape=[1], name="hour")

emb_seq_title = Embedding(vocab_size, EMBEDDING_DIM1, weights = [embedding_matrix1], trainable = False)(seq_title)
emb_seq_domain = Embedding(vocab_size, EMBEDDING_DIM1, weights = [embedding_matrix1], trainable = False)(seq_domain)

rnn_layer1 = GRU(100) (emb_seq_title)
rnn_layer2 = GRU(100) (emb_seq_domain)

rnn_layer3 = Bidirectional(GRU(100, return_sequences = True))(emb_seq_title)
rnn_layer3 = Conv1D(int(100/2), kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(rnn_layer3)
avg_pool1 = GlobalAveragePooling1D()(rnn_layer3)
max_pool1 = GlobalMaxPooling1D()(rnn_layer3)

rnn_layer4 = Bidirectional(GRU(100, return_sequences = True))(emb_seq_domain)
rnn_layer4 = Conv1D(int(100/2), kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(rnn_layer4)
avg_pool2 = GlobalAveragePooling1D()(rnn_layer4)
max_pool2 = GlobalMaxPooling1D()(rnn_layer4)

rnn_layer1 = Dropout(0.2)((Dense(1024, activation='relu')) (rnn_layer1))
rnn_layer1 = BatchNormalization() (rnn_layer1)

rnn_layer2 = Dropout(0.2)((Dense(1024, activation='relu')) (rnn_layer2))
rnn_layer2 = BatchNormalization() (rnn_layer2)
     
o_layer = Dropout(0.2)(Dense(1024, activation='relu') (over_18))
o_layer = BatchNormalization() (o_layer)

t_layer = Dropout(0.2)(Dense(1024, activation='relu') (time_submitted))
t_layer = BatchNormalization() (t_layer)

s_layer = Dropout(0.2)(Dense(1024, activation='relu') (submitter_id))
s_layer = BatchNormalization() (s_layer)

l_layer = Dropout(0.2)(Dense(1024, activation='relu') (title_len))
l_layer = BatchNormalization() (l_layer)

h_layer = Dropout(0.2)(Dense(1024, activation='relu') (hour))
h_layer = BatchNormalization() (h_layer)
               
main_layer = concatenate([
    rnn_layer1,
    rnn_layer2,
    avg_pool1,
    max_pool1,
    avg_pool2,
    max_pool2,
    o_layer,
    t_layer,
    s_layer,
    l_layer,
    h_layer
])

main_layer = Dropout(0.2)(Dense(1024,activation='relu') (main_layer))
main_layer = BatchNormalization() (main_layer)
main_layer = Dropout(0.2)(Dense(512,activation='relu') (main_layer))
main_layer = BatchNormalization() (main_layer)
main_layer = Dropout(0.2)(Dense(128,activation='relu') (main_layer))
main_layer = BatchNormalization() (main_layer)
main_layer = Dropout(0.1)(Dense(64,activation='relu') (main_layer))

output = Dense(1, activation="sigmoid") (main_layer)

model = Model([
            over_18,
            submitter_id,
            time_submitted,
            seq_title,
            seq_domain,
            title_len,
            hour
], output)

from keras import optimizers

adam = optimizers.Adam()

model.compile(optimizer = adam,
            loss= 'binary_crossentropy',
            metrics = ['binary_accuracy'])

model.fit(get_dict(X_train), np.array(Y_train), batch_size = 512*3, validation_split = 0.1, epochs = 10)


# In[ ]:


vals_preds = model.predict(get_dict(test))
Y_pred = vals_preds[:, 0]
Y_pred = np.clip(Y_pred, 0, 1)
Y_pred = (Y_pred >= 0.5)
Y_pred = Y_pred.astype(int)
    
id = np.array(test['id']).astype(int)
my_prediction = pd.DataFrame(Y_pred, id, columns = ['engaging'])

my_prediction.to_csv("my_prediction.csv", index_label = ['id'])

print("The end ...")

