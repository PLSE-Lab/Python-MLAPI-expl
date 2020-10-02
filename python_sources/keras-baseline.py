#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


dftrain = pd.read_csv('../input/train.tsv', delimiter='\t')
dftest = pd.read_csv('../input/test.tsv', delimiter='\t')


# In[ ]:


dftrain.head()


# In[ ]:


len(dftrain)


# In[ ]:


def fillna(df):
    df['category_name'] = df['category_name'].fillna('NaN')
    df['brand_name'] = df['brand_name'].fillna('NaN')
    df['item_description'] = df['item_description'].fillna('NaN')
    return df

dftrain = fillna(dftrain)
dftest = fillna(dftest)


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


category_encoder = LabelEncoder().fit(list(dftrain['category_name']) + list(dftest['category_name']))
dftrain['category_index'] = category_encoder.transform(dftrain['category_name'])
dftest['category_index'] = category_encoder.transform(dftest['category_name'])


# In[ ]:


dftrain.head()


# In[ ]:


brand_encoder = LabelEncoder().fit(list(dftrain['brand_name']) + list(dftest['brand_name']))
dftrain['brand_index'] = brand_encoder.transform(dftrain['brand_name'])
dftest['brand_index'] = brand_encoder.transform(dftest['brand_name'])


# In[ ]:


dftrain.head()


# In[ ]:


from keras.preprocessing.text import Tokenizer


# In[ ]:


name_tokenizer = Tokenizer()
name_tokenizer.fit_on_texts(dftrain['name'])


# In[ ]:


description_tokenizer = Tokenizer()
description_tokenizer.fit_on_texts(dftrain['item_description'])


# In[ ]:


dftrain['seq_name'] = name_tokenizer.texts_to_sequences(dftrain['name'])
dftest['seq_name'] = name_tokenizer.texts_to_sequences(dftest['name'])


# In[ ]:


dftrain['seq_description'] = name_tokenizer.texts_to_sequences(dftrain['item_description'])
dftest['seq_description'] = name_tokenizer.texts_to_sequences(dftest['item_description'])


# In[ ]:


dftrain.head()


# In[ ]:


sum(map(len, dftrain['seq_name'])) / len(dftrain), max(map(len, dftrain['seq_name']))


# In[ ]:


sum(map(len, dftrain['seq_description'])) / len(dftrain), max(map(len, dftrain['seq_description']))


# In[ ]:


from keras.preprocessing.sequence import pad_sequences


# In[ ]:


def padded_sequences(values, maxlen):
    padded = pad_sequences(np.array(values), maxlen=maxlen)
    return [row for row in padded]


# In[ ]:


dftrain['seq_name_padded'] = padded_sequences(dftrain['seq_name'], 10)
dftest['seq_name_padded'] = padded_sequences(dftest['seq_name'], 10)


# In[ ]:


dftrain['seq_description_padded'] = padded_sequences(dftrain['seq_description'], 40)
dftest['seq_description_padded'] = padded_sequences(dftest['seq_description'], 40)


# In[ ]:


dftrain.head()


# In[ ]:


from keras import backend as K


# In[ ]:


def RMSLE(y_true, y_pred):
    y_true_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.0)
    y_pred_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.0)
    return K.sqrt(K.mean(K.square(y_true_log - y_pred_log), 
                         axis=-1))

def RMSE(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 


# In[ ]:


from keras.layers import Input, Embedding, LSTM, concatenate, Dropout, Dense, Flatten
from keras.models import Model


# In[ ]:


len(name_tokenizer.word_index)


# In[ ]:


len(description_tokenizer.word_index)


# In[ ]:


len(category_encoder.classes_)


# In[ ]:


def build_model(name_input_length, description_input_length,
                name_embedding_size, description_embedding_size, 
                brand_embedding_size, category_embedding_size, condition_embedding_size):
    name_input = Input(shape=(name_input_length,), dtype='int32')
    description_input = Input(shape=(description_input_length,), dtype='int32')
    brand_input = Input(shape=(1,), dtype='int32')
    category_input = Input(shape=(1,), dtype='int32')
    condition_input = Input(shape=(1,), dtype='int32')
    shipping_input = Input(shape=(1,), dtype='float32')
    
    name_embedding = Embedding(len(name_tokenizer.word_index) + 1, name_embedding_size)(name_input)
    description_embedding = Embedding(len(description_tokenizer.word_index) + 1, description_embedding_size)(description_input)
    brand_embedding = Embedding(len(brand_encoder.classes_) + 1, brand_embedding_size)(brand_input)
    category_embedding = Embedding(len(category_encoder.classes_) + 1, category_embedding_size)(category_input)
    condition_embedding = Embedding(len(set(dftrain['item_condition_id'])) + 1, condition_embedding_size)(condition_input)
    
    name_rnn = LSTM(8, activation='relu')(name_embedding)
    description_rnn = LSTM(16, activation='relu')(description_embedding)
    
    concat = concatenate([
        name_rnn,
        description_rnn,
        Flatten()(brand_embedding),
        Flatten()(category_embedding),
        Flatten()(condition_embedding),
        shipping_input
    ])
    fc1 = Dropout(0.5)(Dense(128, activation='relu')(concat))
    fc2 = Dropout(0.5)(Dense(64, activation='relu')(fc1))
    fc3 = Dense(1, activation='linear')(fc2)
    
    model = Model([name_input, description_input, brand_input, category_input, condition_input, shipping_input],
                  fc3)
    model.compile(optimizer='rmsprop', loss='mean_squared_logarithmic_error', metrics=[RMSLE])
    return model


# In[ ]:


model = build_model(name_input_length=10,
                    description_input_length=40,
                    name_embedding_size=50,
                    description_embedding_size=50,
                    brand_embedding_size=10,
                    category_embedding_size=10,
                    condition_embedding_size=5)


# In[ ]:


model.summary()


# In[ ]:


X = [np.array([row for row in dftrain['seq_name_padded']]),
     np.array([row for row in dftrain['seq_description_padded']]),
     np.array(dftrain['brand_index']),
     np.array(dftrain['category_index']),
     np.array(dftrain['item_condition_id']),
     np.array(dftrain['shipping']) * 1.0]
y = np.array(dftrain[['price']]) + 1


# In[ ]:


from keras.callbacks import ModelCheckpoint, EarlyStopping


# In[ ]:


model.fit(X, y, epochs=5, validation_split=0.1, batch_size=20000, callbacks=[
    ModelCheckpoint('checkpoint.hdf5', save_best_only=True),
    EarlyStopping(patience=5),
])


# In[ ]:


model.fit(X, y, epochs=5, validation_split=0.1, batch_size=20000, callbacks=[
    ModelCheckpoint('checkpoint.hdf5', save_best_only=True),
    EarlyStopping(patience=5),
])


# In[ ]:


model.fit(X, y, epochs=5, validation_split=0.1, batch_size=20000, callbacks=[
    ModelCheckpoint('checkpoint.hdf5', save_best_only=True),
    EarlyStopping(patience=5),
])


# In[ ]:


model.fit(X, y, epochs=5, validation_split=0.1, batch_size=20000, callbacks=[
    ModelCheckpoint('checkpoint.hdf5', save_best_only=True),
    EarlyStopping(patience=5),
])


# In[ ]:


model.load_weights('checkpoint.hdf5')


# In[ ]:


X = [np.array([row for row in dftest['seq_name_padded']]),
     np.array([row for row in dftest['seq_description_padded']]),
     np.array(dftest['brand_index']),
     np.array(dftest['category_index']),
     np.array(dftest['item_condition_id']),
     np.array(dftest['shipping']) * 1.0]
prediction = model.predict(X, verbose=True, batch_size=20000)[:, 0]


# In[ ]:


from collections import OrderedDict

df = pd.DataFrame(OrderedDict([
    ('test_id', dftest['test_id']),
    ('price', prediction)
]))
df.head()


# In[ ]:


df.to_csv('submission.csv', index=None)


# In[ ]:




