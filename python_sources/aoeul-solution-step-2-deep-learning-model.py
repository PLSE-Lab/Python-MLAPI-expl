#!/usr/bin/env python
# coding: utf-8

# The neural network architecture is based on https://www.kaggle.com/artgor/movie-review-sentiment-analysis-eda-and-models 
# 
# Actually this model is an overkill for this competition and takes too long (8 hours!) to train and predict. This is what happened when you copied someone's code without knowing (or assumed wrongly) what you were actually doing.
# 
# Team yellow's simple feedforward neural network (https://www.kaggle.com/astraldawn/yellow-feedforward-neural-network) is much better in terms of speed and performance.

# # Import

# In[ ]:


from datetime import datetime
start = datetime.now()


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from pathlib import Path
import json
import re
import sys
import warnings


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from sklearn.model_selection import train_test_split


# In[ ]:


import tensorflow as tf
from tensorflow import keras

from keras.preprocessing import text, sequence
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Embedding, BatchNormalization, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Input, GlobalMaxPool1D, Conv1D, MaxPooling1D, GRU, concatenate, CuDNNGRU
from keras.layers import LSTM, CuDNNLSTM,  Bidirectional, SpatialDropout1D
from keras import optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
from keras import utils
from keras.utils import to_categorical
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.callbacks import Callback
from keras.optimizers import Adam

keras.__version__, tf.__version__


# In[ ]:


VERSION = 'Deep_Learning_predictions'


# In[ ]:


DATA_DIR = Path('../input')

BEAUTY_JSON = DATA_DIR / 'beauty_profile_train.json'
FASHION_JSON = DATA_DIR / 'fashion_profile_train.json'
MOBILE_JSON = DATA_DIR / 'mobile_profile_train.json'

BEAUTY_TRAIN_CSV = DATA_DIR / 'beauty_data_info_train_competition.csv'
FASHION_TRAIN_CSV = DATA_DIR / 'fashion_data_info_train_competition.csv'
MOBILE_TRAIN_CSV = DATA_DIR / 'mobile_data_info_train_competition.csv'

BEAUTY_TEST_CSV = DATA_DIR / 'beauty_data_info_val_competition.csv'
FASHION_TEST_CSV = DATA_DIR / 'fashion_data_info_val_competition.csv'
MOBILE_TEST_CSV = DATA_DIR / 'mobile_data_info_val_competition.csv'


# In[ ]:


with open(BEAUTY_JSON) as f:
     beauty_attribs = json.load(f)
        
with open(FASHION_JSON) as f:
     fashion_attribs = json.load(f)
        
with open(MOBILE_JSON) as f:
     mobile_attribs = json.load(f)

beauty_train_df = pd.read_csv(BEAUTY_TRAIN_CSV)
fashion_train_df = pd.read_csv(FASHION_TRAIN_CSV)
mobile_train_df = pd.read_csv(MOBILE_TRAIN_CSV)

beauty_test_df = pd.read_csv(BEAUTY_TEST_CSV)
fashion_test_df = pd.read_csv(FASHION_TEST_CSV)
mobile_test_df = pd.read_csv(MOBILE_TEST_CSV)


# In[ ]:


len(beauty_train_df), len(fashion_train_df), len(mobile_train_df)


# In[ ]:


len(beauty_test_df), len(fashion_test_df), len(mobile_test_df)


# In[ ]:


n_rows = len(beauty_test_df)*5 + len(fashion_test_df)*5 + len(mobile_test_df)*11 
print(n_rows)
assert n_rows == 977987, "Row numbers don't match!"


# # Hyperparameters

# In[ ]:


max_words_beauty = 29
max_words_fashion = 32
max_words_mobile = 27
batch_size = 32
epochs_beauty = [25, 25, 25, 25, 25]
epochs_fashion = [20, 20, 20, 20, 20]
epochs_mobile = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
embed_size = 256


# # Models

# In[ ]:


def build_model(max_features, max_len, num_classes, lr=0.0, lr_d=0.0, units=0, spatial_dr=0.0, kernel_size1=3, 
                 kernel_size2=2, dense_units=128, dr=0.1, conv_size=32):
    inp = Input(shape=(max_len,))
    x = Embedding(max_features, embed_size)(inp)
    x1 = SpatialDropout1D(spatial_dr)(x)

    x_gru = Bidirectional(CuDNNGRU(units, return_sequences=True))(x1)
    x_lstm = Bidirectional(CuDNNLSTM(units, return_sequences=True))(x1)

    x_conv1 = Conv1D(conv_size, kernel_size=kernel_size1, padding='valid', kernel_initializer='he_uniform')(x_gru)
    avg_pool1_gru = GlobalAveragePooling1D()(x_conv1)
    max_pool1_gru = GlobalMaxPooling1D()(x_conv1)

    x_conv2 = Conv1D(conv_size, kernel_size=kernel_size2, padding='valid', kernel_initializer='he_uniform')(x_gru)
    avg_pool2_gru = GlobalAveragePooling1D()(x_conv2)
    max_pool2_gru = GlobalMaxPooling1D()(x_conv2)

    x_conv3 = Conv1D(conv_size, kernel_size=kernel_size1, padding='valid', kernel_initializer='he_uniform')(x_lstm)
    avg_pool1_lstm = GlobalAveragePooling1D()(x_conv3)
    max_pool1_lstm = GlobalMaxPooling1D()(x_conv3)

    x_conv4 = Conv1D(conv_size, kernel_size=kernel_size2, padding='valid', kernel_initializer='he_uniform')(x_lstm)
    avg_pool2_lstm = GlobalAveragePooling1D()(x_conv4)
    max_pool2_lstm = GlobalMaxPooling1D()(x_conv4)

    x = concatenate([avg_pool1_gru, max_pool1_gru, avg_pool2_gru, max_pool2_gru,
                     avg_pool1_lstm, max_pool1_lstm, avg_pool2_lstm, max_pool2_lstm])
    x = BatchNormalization()(x)
    x = Dropout(dr)(Dense(dense_units, activation='relu')(x))
    x = BatchNormalization()(x)
    x = Dropout(dr)(Dense(int(dense_units/2), activation='relu')(x))
    x = Dense(units=num_classes, activation='softmax')(x)
    
    model = Model(inputs=inp, outputs=x)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])

    return model


# In[ ]:


def RNN_model(max_features, num_classes, maxlen):    
    model = build_model(max_features, maxlen, num_classes,
                         lr = 1e-4, lr_d = 0, units = 64, spatial_dr = 0.5, 
                         kernel_size1=4, kernel_size2=3, dense_units=32, dr=0.1, conv_size=32)
    return model


# # Training

# ## Beauty

# In[ ]:


print(f'Total attributes for beauty products: {len(beauty_attribs)}')
print()

for i, attrib in enumerate(beauty_attribs, 1):
    print(f'Attribute_{i}: {attrib} ({len(beauty_attribs[attrib])} classes)')

all_beauty_titles = list(beauty_train_df.title)
beauty_tokenizer = Tokenizer()
beauty_tokenizer.fit_on_texts(all_beauty_titles)
tokenized_train_titles = beauty_tokenizer.texts_to_sequences(list(beauty_train_df.title))
print()
print(f'Total {len(beauty_tokenizer.word_index)} unique tokens.')


# In[ ]:


map_beauty = []
for i, attrib in enumerate(beauty_attribs):
    num_classes = len(beauty_attribs[attrib])
    print(f'[Attrib_{i+1} {attrib}]: ({num_classes} classes)')
    beauty_attrib_train_df = beauty_train_df[['title', attrib]].dropna()
    titles = list(beauty_attrib_train_df.title)
    tokenized_train_titles = beauty_tokenizer.texts_to_sequences(titles)

    X = np.array(tokenized_train_titles)
    X = sequence.pad_sequences(X, padding='post', maxlen=max_words_beauty)
    y = to_categorical(beauty_attrib_train_df[attrib], num_classes)
    
    max_features = len(beauty_tokenizer.word_index) # how many unique words to use
    model = RNN_model(max_features, num_classes, max_words_beauty)
    
    file_path = f"{VERSION}_model_beauty_{attrib}.hdf5"
    
    hist = model.fit(X, y, 
                    batch_size=batch_size,
                    epochs=epochs_beauty[i],
                    verbose=0)
    model.save(file_path)
    print()


# In[ ]:


duration = datetime.now()-start
print('Total seconds:', duration.total_seconds())
print(f'Total minutes: {duration.total_seconds()/60:.2f}')
print(f'Total hour: {duration.total_seconds()/3600:.2f}')


# ## Fashion

# In[ ]:


print(f'Total attributes for fashion products: {len(fashion_attribs)}')
print()

for i, attrib in enumerate(fashion_attribs, 1):
    print(f'Attribute_{i}: {attrib} ({len(fashion_attribs[attrib])} classes)')

all_fashion_titles = list(fashion_train_df.title)
fashion_tokenizer = Tokenizer()
fashion_tokenizer.fit_on_texts(all_fashion_titles)
tokenized_train_titles = fashion_tokenizer.texts_to_sequences(list(fashion_train_df.title))
print()
print(f'Total {len(fashion_tokenizer.word_index)} unique tokens.')


# In[ ]:


map_fashion = []
for i, attrib in enumerate(fashion_attribs):
    num_classes = len(fashion_attribs[attrib])
    print(f'[Attrib_{i+1} {attrib}]: ({num_classes} classes)')
    fashion_attrib_train_df = fashion_train_df[['title', attrib]].dropna()
    titles = list(fashion_attrib_train_df.title)
    tokenized_train_titles = fashion_tokenizer.texts_to_sequences(titles)

    X = np.array(tokenized_train_titles)
    X = sequence.pad_sequences(X, padding='post', maxlen=max_words_fashion)
    y = to_categorical(fashion_attrib_train_df[attrib], num_classes)
    
    max_features = len(fashion_tokenizer.word_index) # how many unique words to use
    model = RNN_model(max_features, num_classes, max_words_fashion)
    
    file_path = f"{VERSION}_model_fashion_{attrib}.hdf5"
    
    hist = model.fit(X, y, 
                    batch_size=batch_size,
                    epochs=epochs_fashion[i],
                    verbose=0)
    model.save(file_path)
    print()


# In[ ]:


duration = datetime.now()-start
print('Total seconds:', duration.total_seconds())
print(f'Total minutes: {duration.total_seconds()/60:.2f}')
print(f'Total hour: {duration.total_seconds()/3600:.2f}')


# ## Mobile

# In[ ]:


print(f'Total attributes for mobile products: {len(mobile_attribs)}')
print()

for i, attrib in enumerate(mobile_attribs, 1):
    print(f'Attribute_{i}: {attrib} ({len(mobile_attribs[attrib])} classes)')

all_mobile_titles = list(mobile_train_df.title)
mobile_tokenizer = Tokenizer()
mobile_tokenizer.fit_on_texts(all_mobile_titles)
tokenized_train_titles = mobile_tokenizer.texts_to_sequences(list(mobile_train_df.title))
print()
print(f'Total {len(mobile_tokenizer.word_index)} unique tokens.')


# In[ ]:


map_mobile = []
for i, attrib in enumerate(mobile_attribs):
    num_classes = len(mobile_attribs[attrib])
    print(f'[Attrib_{i+1} {attrib}]: ({num_classes} classes)')
    mobile_attrib_train_df = mobile_train_df[['title', attrib]].dropna()
    titles = list(mobile_attrib_train_df.title)
    tokenized_train_titles = mobile_tokenizer.texts_to_sequences(titles)

    X = np.array(tokenized_train_titles)
    X = sequence.pad_sequences(X, padding='post', maxlen=max_words_mobile)
    y = to_categorical(mobile_attrib_train_df[attrib], num_classes)
    
    max_features = len(mobile_tokenizer.word_index) # how many unique words to use
    model = RNN_model(max_features, num_classes, max_words_mobile)
    
    file_path = f"{VERSION}_model_mobile_{attrib}.hdf5"
    
    hist = model.fit(X, y, 
                    batch_size=batch_size,
                    epochs=epochs_mobile[i],
                    verbose=0)
    model.save(file_path)
    print()


# In[ ]:


duration = datetime.now()-start
print('Total seconds:', duration.total_seconds())
print(f'Total minutes: {duration.total_seconds()/60:.2f}')
print(f'Total hour: {duration.total_seconds()/3600:.2f}')


# # Inference

# In[ ]:


get_ipython().run_cell_magic('time', '', "updated_beauty_title = list(beauty_test_df.title)\ntokenized_test_titles = beauty_tokenizer.texts_to_sequences(updated_beauty_title)\nX_test = np.array(tokenized_test_titles)\nX_test = sequence.pad_sequences(X_test, padding='post', maxlen=max_words_beauty)\n\npreds_all={}\nfor attrib in beauty_attribs:\n    print(f'Attribute: {attrib}')\n    print(f'Loading model...')\n    model=load_model(f'{VERSION}_model_beauty_{attrib}.hdf5')\n    print(f'Predicting...')\n    pred_attrib=model.predict(X_test)\n    print(f'Sorting predictions...')\n    pred_list=[]\n    for pred in pred_attrib:\n        pred_list.append(pred.argsort()[-2:][::-1])\n    preds_all[attrib]=np.array(pred_list.copy())\n    print()\n    \nprint(f'Saving to {VERSION}_pred_beauty.csv')\ntest_y_id=[] \ntest_y_predictions=[]\nfor i, itemid in enumerate(beauty_test_df.itemid):\n    for attrib in beauty_attribs:\n        test_y_id.append(str(itemid) + f'_{attrib}')\n        test_y_predictions.append(str(preds_all[attrib][i][0]) + ' ' + str(preds_all[attrib][i][1]))\n        \nbeauty_result_df = pd.DataFrame(\n    {'id': test_y_id, 'tagging': test_y_predictions},\n    columns = ['id', 'tagging'])\n\nbeauty_result_df.to_csv(f'{VERSION}_pred_beauty.csv', index=False)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "updated_fashion_title = list(fashion_test_df.title)\ntokenized_test_titles = fashion_tokenizer.texts_to_sequences(updated_fashion_title)\nX_test = np.array(tokenized_test_titles)\nX_test = sequence.pad_sequences(X_test, padding='post', maxlen=max_words_fashion)\n\npreds_all={}\nfor attrib in fashion_attribs:\n    print(f'Attribute: {attrib}')\n    print(f'Loading model...')\n    model=load_model(f'{VERSION}_model_fashion_{attrib}.hdf5')\n    print(f'Predicting...')\n    pred_attrib=model.predict(X_test)\n    print(f'Sorting predictions...')\n    pred_list=[]\n    for pred in pred_attrib:\n        pred_list.append(pred.argsort()[-2:][::-1])\n    preds_all[attrib]=np.array(pred_list.copy())\n    print()\n    \nprint(f'Saving to {VERSION}_pred_fashion.csv')\ntest_y_id=[] \ntest_y_predictions=[]\nfor i, itemid in enumerate(fashion_test_df.itemid):\n    for attrib in fashion_attribs:\n        test_y_id.append(str(itemid) + f'_{attrib}')\n        test_y_predictions.append(str(preds_all[attrib][i][0]) + ' ' + str(preds_all[attrib][i][1]))\n        \nfashion_result_df = pd.DataFrame(\n    {'id': test_y_id, 'tagging': test_y_predictions},\n    columns = ['id', 'tagging'])\n\nfashion_result_df.to_csv(f'{VERSION}_pred_fashion.csv', index=False)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "updated_mobile_title = list(mobile_test_df.title)\ntokenized_test_titles = mobile_tokenizer.texts_to_sequences(updated_mobile_title)\nX_test = np.array(tokenized_test_titles)\nX_test = sequence.pad_sequences(X_test, padding='post', maxlen=max_words_mobile)\n\npreds_all={}\nfor attrib in mobile_attribs:\n    print(f'Attribute: {attrib}')\n    print(f'Loading model...')\n    model=load_model(f'{VERSION}_model_mobile_{attrib}.hdf5')\n    print(f'Predicting...')\n    pred_attrib=model.predict(X_test)\n    print(f'Sorting predictions...')\n    pred_list=[]\n    for pred in pred_attrib:\n        pred_list.append(pred.argsort()[-2:][::-1])\n    preds_all[attrib]=np.array(pred_list.copy())\n    print()\n    \nprint(f'Saving to {VERSION}_pred_mobile.csv')\ntest_y_id=[] \ntest_y_predictions=[]\nfor i, itemid in enumerate(mobile_test_df.itemid):\n    for attrib in mobile_attribs:\n        test_y_id.append(str(itemid) + f'_{attrib}')\n        test_y_predictions.append(str(preds_all[attrib][i][0]) + ' ' + str(preds_all[attrib][i][1]))\n        \nmobile_result_df = pd.DataFrame(\n    {'id': test_y_id, 'tagging': test_y_predictions},\n    columns = ['id', 'tagging'])\n\nmobile_result_df.to_csv(f'{VERSION}_pred_mobile.csv', index=False)")


# In[ ]:


combined_df = pd.concat([beauty_result_df, fashion_result_df, mobile_result_df], axis=0)
combined_df.to_csv(f'{VERSION}.csv', index=None)


# In[ ]:


duration = datetime.now()-start
print('Total seconds:', duration.total_seconds())
print(f'Total minutes: {duration.total_seconds()/60:.2f}')
print(f'Total hour: {duration.total_seconds()/3600:.2f}')

