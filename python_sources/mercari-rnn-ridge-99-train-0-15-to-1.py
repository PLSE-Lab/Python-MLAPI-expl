#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import math

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import FeatureUnion

from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import Ridge, LogisticRegression

import keras
import keras.backend as K

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, LSTM, GRU, Dropout, Flatten, Embedding, Dense, concatenate, Reshape

pd.set_option('display.float_format', lambda x: '%.3f' % x)


# In[ ]:


import pyximport
pyximport.install()
import os
import random
import numpy as np
import tensorflow as tf
os.environ['PYTHONHASHSEED'] = '10000'
np.random.seed(10001)
random.seed(10002)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=5, inter_op_parallelism_threads=1)
from keras import backend
tf.set_random_seed(10003)
backend.set_session(tf.Session(graph=tf.get_default_graph(), config=session_conf))


# In[ ]:


def rmsle_cust(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))

def rmsle_log(Y, Y_pred):
    assert Y.shape == Y_pred.shape
    return np.sqrt(np.mean(np.square(Y_pred - Y )))

def rmsle(h, y): 
    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())

def load_data(path, sep=','):
    train = pd.read_csv(path, sep='\t')    
    return train
    
def preprocess_rnn(train_data, test_data):
    
    print('Dropping extra columns and splitting data into x and y')
    y_train = train_data['price']    
    x_train = train.drop(['train_id', 'price'], axis=1)
    x_test = test.drop(['test_id'], axis=1)

    print('Filling in missing values')
    x_train.item_description.fillna(value="missing", inplace=True)
    x_train.brand_name.fillna(value="missing", inplace=True)
    x_train.category_name.fillna(value='missing', inplace=True)

    x_test.item_description.fillna(value="missing", inplace=True)
    x_test.brand_name.fillna(value="missing", inplace=True)
    x_test.category_name.fillna(value='missing', inplace=True)

    print('Transforming categorical values to lowercase')
    train_name = x_train['name'].str.lower().values
    train_brand_name = x_train['brand_name'].str.lower().values
    train_item_description = x_train['item_description'].str.lower().values
    train_category_name = x_train['category_name'].str.lower().values
    train_item_condition_id = x_train['item_condition_id'].values
    train_shipping = x_train['shipping'].values

    test_name = x_test['name'].str.lower().values
    test_brand_name = x_test['brand_name'].str.lower().values
    test_item_description = x_test['item_description'].str.lower().values
    test_category_name = x_test['category_name'].str.lower().values
    test_item_condition_id = x_test['item_condition_id'].values
    test_shipping = x_test['shipping'].values

    print('Fitting the Tokenizer on the categorical data')
    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(np.concatenate([train_name, train_brand_name, train_category_name, 
                                          test_name, test_brand_name, test_category_name]))        
    
    print('Transforming categorical data using the tokenizr')
    train_name = tokenizer.texts_to_sequences(train_name)
    train_brand_name = tokenizer.texts_to_sequences(train_brand_name)
    train_item_description = tokenizer.texts_to_sequences(train_item_description)
    train_category_name = tokenizer.texts_to_sequences(train_category_name)

    test_name = tokenizer.texts_to_sequences(test_name)
    test_brand_name = tokenizer.texts_to_sequences(test_brand_name)
    test_item_description = tokenizer.texts_to_sequences(test_item_description)
    test_category_name = tokenizer.texts_to_sequences(test_category_name)

    print('Split data into train and val')

    name_train, name_val, brand_name_train, brand_name_val, item_description_train,     item_description_val, category_name_train, category_name_val, item_condition_id_train,     item_condition_id_val, shipping_train, shipping_val, y_train, y_val =     train_test_split(train_name, train_brand_name, train_item_description, train_category_name,                      train_item_condition_id, train_shipping, y_train, test_size=0.01)

    print('Padding sequences to set length')
    x_train = {
        'name': pad_sequences(name_train, 5),
        'item_description': pad_sequences(item_description_train, 25),
        'brand_name': pad_sequences(brand_name_train, 2),
        'category_name': pad_sequences(category_name_train, 2),
        'item_condition_id': np.array(item_condition_id_train),
        'shipping': np.array(shipping_train)
    }
    
    x_val = {
        'name': pad_sequences(name_val, 5),
        'item_description': pad_sequences(item_description_val, 25),
        'brand_name': pad_sequences(brand_name_val, 2),
        'category_name': pad_sequences(category_name_val, 2),
        'item_condition_id': np.array(item_condition_id_val),
        'shipping': np.array(shipping_val)
    }

    x_test = {
        'name': pad_sequences(test_name, 5),
        'item_description': pad_sequences(test_item_description, 25),
        'brand_name': pad_sequences(test_brand_name, 2),
        'category_name': pad_sequences(test_category_name, 2),
        'item_condition_id': np.array(test_item_condition_id),
        'shipping': np.array(test_shipping)
    }
    return x_train, y_train, x_val, y_val, x_test

def preprocess_ridge(train_data, test_data,):
    
    print('Dropping extra columns and splitting data into x and y')
    y_train = train_data['price']
    x_train = train.drop(['train_id', 'price'], axis=1)
    x_test = test.drop(['test_id'], axis=1)
    
    print('Filling in missing values')
    x_train.item_description.fillna(value="missing", inplace=True)
    x_train.brand_name.fillna(value="missing", inplace=True)
    x_train.category_name.fillna(value='missing', inplace=True)

    x_test.item_description.fillna(value="missing", inplace=True)
    x_test.brand_name.fillna(value="missing", inplace=True)
    x_test.category_name.fillna(value='missing', inplace=True)
    
    print('Use label encoder on categorical data')
    le = LabelEncoder()    
    le.fit(np.concatenate([x_train['category_name'], x_test['category_name']]))
    x_train['category_name'] = le.transform(x_train['category_name'])
    x_test['category_name'] = le.transform(x_test['category_name'])

    le = LabelEncoder()
    le.fit(np.concatenate([x_train['brand_name'], x_test['brand_name']]))
    x_train['brand_name'] = le.transform(x_train['brand_name']).astype('str')
    x_test['brand_name'] = le.transform(x_test['brand_name']).astype('str')
            
    print('Split data into train and val')
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.01)
    print(len(y_train), len(x_train), len(x_val), len(x_test))

    return x_train, y_train, x_val, y_val, x_test

def predict_rnn(model, data):
    preds = model.predict(data, verbose=True)
    preds = np.abs(preds)

    return preds

def predict_ridge(model, data):
    preds = model.predict(data)
    preds = np.abs(preds)

    return preds

def submit(preds):
    index = pd.read_csv(f'../input/test.tsv', sep='\t')['test_id'].values
    submission = pd.DataFrame(index, columns=['test_id'], index=None)    
    submission['price'] = preds
    submission.to_csv('submission.csv',index=False)


# In[ ]:


get_ipython().run_cell_magic('time', '', "train = load_data('../input/train.tsv', '\\t')\ntest = load_data('../input/test.tsv', '\\t')\nx_train_rnn, y_train_rnn, x_val_rnn, y_val_rnn, x_test_rnn = preprocess_rnn(train, test)\nx_train_ridge, y_train_ridge, x_val_ridge, y_val_ridge, x_test_ridge = preprocess_ridge(train, test)")


# In[ ]:


y_train_rnn = np.log1p(y_train_rnn)
y_val_rnn = np.log1p(y_val_rnn)


# In[ ]:


name_input = Input(shape=[x_train_rnn["name"].shape[1]], name='name')
item_description_input = Input(shape=[25], name='item_description')
brand_name_input = Input(shape=[2], name='brand_name')
category_name_input = Input(shape=[2], name='category_name')
item_condition_id_input = Input(shape=[1], name='item_condition_id')
shipping_input = Input(shape=[1], name='shipping')

name_embedding = Embedding(1000, 20)(name_input)
item_description_embedding = Embedding(1000, 75)(item_description_input)
brand_name_embedding = Embedding(1000, 10)(brand_name_input)
category_name_embedding = Embedding(1000, 10)(category_name_input)

name_rnn = GRU(8)(name_embedding)
item_description_rnn = GRU(16)(item_description_embedding)

concatenate_1 = concatenate([
    (name_rnn),
    (item_description_rnn),
    Flatten()(brand_name_embedding),
    Flatten()(category_name_embedding),
    (item_condition_id_input),
    (shipping_input),    
])

dense_1 = Dense(256, activation='relu')(concatenate_1)
dropout_1 = Dropout(0.2)(dense_1)
dense_2 = Dense(128, activation='relu')(dropout_1)
dropout_2 = Dropout(0.2)(dense_2)
dense_3 = Dense(64, activation='relu')(dropout_2)
dropout_3 = Dropout(0.2)(dense_3)

output = Dense(1, activation="linear")(dropout_3)

rnn_model = Model([name_input, brand_name_input, item_description_input, category_name_input, 
               item_condition_id_input, shipping_input], output)

rnn_model.compile(loss="mse", optimizer=keras.optimizers.Adam(), metrics=['mae', rmsle_cust])
rnn_model.summary()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'rnn_model.fit(x_train_rnn, y_train_rnn, batch_size=64, validation_data=(x_val_rnn, y_val_rnn), epochs=1)')


# In[ ]:


rnn_val_preds = rnn_model.predict(x_val_rnn, verbose=1)
rnn_train_preds = rnn_model.predict(x_train_rnn, verbose=1)


# ## RNN Metrics

# ### RMSLE

# In[ ]:


rmsle(np.expm1(rnn_train_preds), np.expm1(y_train_rnn.values.reshape(-1, 1)))


# In[ ]:


rmsle(np.expm1(rnn_val_preds), np.expm1(y_val_rnn.values.reshape(-1, 1)))


# ### MAE

# In[ ]:


mean_absolute_error(np.expm1(y_train_rnn), np.expm1(rnn_train_preds))


# In[ ]:


mean_absolute_error(np.expm1(y_val_rnn), np.expm1(rnn_val_preds))


# In[ ]:


full_df = pd.concat([x_train_ridge, x_val_ridge, x_test_ridge])
full_df['shipping'] = full_df['shipping'].astype(str)
full_df['item_condition_id'] = full_df['item_condition_id'].astype(str)
full_df['category_name'] = full_df['category_name'].astype(str)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'print("Vectorizing data...")\ndefault_preprocessor = CountVectorizer().build_preprocessor()\ndef build_preprocessor(field):\n    field_idx = list(full_df.columns).index(field)\n    return lambda x: default_preprocessor(x[field_idx])\n\nvectorizer = FeatureUnion([\n    (\'name\', CountVectorizer(\n        ngram_range=(1, 2),\n        max_features=50000,\n        preprocessor=build_preprocessor(\'name\'))),\n    (\'category_name\', CountVectorizer(\n        token_pattern=\'.+\',\n        preprocessor=build_preprocessor(\'category_name\'))),\n    (\'brand_name\', CountVectorizer(\n        token_pattern=\'.+\',\n        preprocessor=build_preprocessor(\'brand_name\'))),\n    (\'shipping\', CountVectorizer(\n        token_pattern=\'\\d+\',\n        preprocessor=build_preprocessor(\'shipping\'))),\n    (\'item_condition_id\', CountVectorizer(\n        token_pattern=\'\\d+\',\n        preprocessor=build_preprocessor(\'item_condition_id\'))),\n    (\'item_description\', TfidfVectorizer(\n        ngram_range=(1, 3),\n        max_features=100000,\n        preprocessor=build_preprocessor(\'item_description\'))),\n])\n\nX = vectorizer.fit_transform(full_df.values)\n\nX_train_ridge = X[:1467709]\nX_dev_ridge = X[1467709:14826 + 1467709]\nX_test_ridge = X[14826 + 1467709:]\n\nprint(X.shape, X_train_ridge.shape, X_dev_ridge.shape, X_test_ridge.shape)')


# In[ ]:


y_train_ridge = np.log1p(y_train_ridge)
y_val_ridge = np.log1p(y_val_ridge)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'ridge_model = Ridge()\nridge_model.fit(X_train_ridge, y_train_ridge)')


# ## Ridge Metrics

# ### RMSLE

# In[ ]:


train_ridge_preds = predict_ridge(ridge_model, X_train_ridge)
val_ridge_preds = predict_ridge(ridge_model, X_dev_ridge)


# In[ ]:


rmsle(np.expm1(train_ridge_preds), np.expm1(y_train_ridge))


# In[ ]:


rmsle(np.expm1(val_ridge_preds), np.expm1(y_val_ridge))


# In[ ]:


mean_absolute_error(np.expm1(y_train_ridge), np.expm1(train_ridge_preds))


# In[ ]:


mean_absolute_error(np.expm1(y_val_ridge), np.expm1(val_ridge_preds))


# In[ ]:


def aggregate_predicts(Y1, Y2, ratio):
    assert Y1.shape == Y2.shape
    return Y1 * ratio + Y2 * (1.0 - ratio)


# ## Metrics

# ### Aggregate

# In[ ]:


val_ridge_preds = np.expm1(val_ridge_preds)
train_ridge_preds = np.expm1(train_ridge_preds)


# In[ ]:


rnn_val_preds = np.expm1(rnn_val_preds)
rnn_train_preds = np.expm1(rnn_train_preds)


# In[ ]:


aggregate_val = aggregate_predicts(rnn_val_preds, val_ridge_preds.reshape(-1, 1), 0.15)
aggregate_train = aggregate_predicts(rnn_train_preds, train_ridge_preds.reshape(-1, 1), 0.15)


# ### RMSLE

# In[ ]:


rmsle(aggregate_val, np.expm1(y_val_rnn.values.reshape(-1, 1)))


# In[ ]:


rmsle(aggregate_train, np.expm1(y_train_rnn.values.reshape(-1, 1)))


# ### MAE

# In[ ]:


mean_absolute_error(np.expm1(y_val_rnn.values.reshape(-1, 1)), aggregate_val)


# In[ ]:


mean_absolute_error(np.expm1(y_train_rnn.values.reshape(-1, 1)), aggregate_train)


# ## Predict

# In[ ]:


get_ipython().run_cell_magic('time', '', 'rnn_preds = predict_rnn(rnn_model, x_test_rnn)\nrnn_preds = np.expm1(rnn_preds)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'ridge_preds = predict_ridge(ridge_model, X_test_ridge)\nridge_preds = np.expm1(ridge_preds)')


# In[ ]:


total_preds = aggregate_predicts(rnn_preds, ridge_preds.reshape(-1, 1), 0.15)


# In[ ]:


total_preds.shape


# In[ ]:


submit(total_preds)


# In[ ]:


pd.read_csv('submission.csv')

