#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model,Model
from keras.layers import Dense, LSTM, Dropout, Embedding, Input, Bidirectional
from keras.layers import SimpleRNNCell, Concatenate, Add, RNN, BatchNormalization
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from scipy.stats import spearmanr
from keras.initializers import RandomNormal, RandomUniform
from keras import regularizers
import time
from pathlib import Path
from tqdm import tqdm

def clean_texts(text):
    
    puncts = '.,?/\\!-:+#@()*%$\'"'
    
    stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
                 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
                 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
                 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 
                 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
                 'or', 'as', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
                 'into', 'through', 'above', 'to', 'from', 'up', 'in', 'out',
                 'on', 'off', 'under', 'again', 'further', 'then', 'once', 'here',
                 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
                 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only',
                 'own', 'so', 'than', 'too', 'very', 'will', 'just', 'should']
    
    for word in stopwords:
        if word in text:
            text = text.replace(word, '')
    for punct in puncts:
        if punct in text:
            text = text.replace(punct, '')
    
    return text



def cat_to_numeric(category):
    if category=='LIFE_ARTS':
        return 1
    if category=='CULTURE':
        return 2
    if category=='SCIENCE':
        return 3
    if category=='STACKOVERFLOW':
        return 4
    if category=='TECHNOLOGY':
        return 5

def prepare_data(frame):
    
    to_drop = []
    for col in frame.columns:
        if 'user_page' in col or 'host' in col or        'url' in col or 'user_name' in col or 'categ' in col:
            to_drop.append(col)

    data = frame.drop(to_drop, axis=1)
    return data


def get_vars_and_targets(train_data, test_data):
    
    train_cols = test_data.columns
    target_cols = set(train_data.columns).difference(set(test_data.columns))
    train_cols = set(train_data.columns) - target_cols
    target_cols = list(target_cols)
    train_cols = list(train_cols)
    return train_cols, target_cols

def get_text_cols(frame):
    
    text_cols = []
    for col in frame.columns:
        if 'title' in col or 'body' in col or col=='answer':
            text_cols.append(col)
            
    return text_cols


def transform_texts(frame, tokenizer=None, test=False):
    
    text_cols = get_text_cols(frame)
    for ind in frame.index:
        for col in text_cols:
            frame.loc[ind, col] = clean_texts(str(frame.loc[ind, col]))
    if test==False:
        tokenizer = Tokenizer(oov_token='OOV')
        for col in text_cols:
            tokenizer.fit_on_texts(frame[col])
    else:
        tokenizer=tokenizer
    renamed_cols = []
    for col in text_cols:
        renamed_cols.append(col+'_tokenized')
        frame[col+'_tokenized'] = tokenizer.texts_to_sequences(frame[col])
        frame.drop([col], inplace=True, axis=1)
    return [frame, renamed_cols, tokenizer]




import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model,Model
from keras.layers import Dense, LSTM, Dropout, Embedding, Input, Bidirectional
from keras.layers import SimpleRNNCell, Concatenate, Add, RNN, BatchNormalization
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from scipy.stats import spearmanr
from keras.initializers import RandomNormal, RandomUniform
from keras import regularizers
import time
from pathlib import Path
from tqdm import tqdm

def clean_texts(text):
    
    puncts = '.,?/\\!-:+#@()*%$\'"'
    
    stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
                 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
                 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
                 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 
                 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
                 'or', 'as', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
                 'into', 'through', 'above', 'to', 'from', 'up', 'in', 'out',
                 'on', 'off', 'under', 'again', 'further', 'then', 'once', 'here',
                 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
                 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only',
                 'own', 'so', 'than', 'too', 'very', 'will', 'just', 'should']
    
    for word in stopwords:
        if word in text:
            text = text.replace(word, '')
    for punct in puncts:
        if punct in text:
            text = text.replace(punct, '')
    
    return text



def cat_to_numeric(category):
    if category=='LIFE_ARTS':
        return 1
    if category=='CULTURE':
        return 2
    if category=='SCIENCE':
        return 3
    if category=='STACKOVERFLOW':
        return 4
    if category=='TECHNOLOGY':
        return 5

def prepare_data(frame):
    
    to_drop = []
    for col in frame.columns:
        if 'user_page' in col or 'host' in col or        'url' in col or 'user_name' in col or 'categ' in col:
            to_drop.append(col)

    data = frame.drop(to_drop, axis=1)
    return data


def get_vars_and_targets(train_data, test_data):
    
    train_cols = test_data.columns
    target_cols = set(train_data.columns).difference(set(test_data.columns))
    train_cols = set(train_data.columns) - target_cols
    target_cols = list(target_cols)
    train_cols = list(train_cols)
    return train_cols, target_cols

def get_text_cols(frame):
    
    text_cols = []
    for col in frame.columns:
        if 'title' in col or 'body' in col or col=='answer':
            text_cols.append(col)
            
    return text_cols


def transform_texts(frame, tokenizer=None, test=False):
    
    text_cols = get_text_cols(frame)
    for ind in frame.index:
        for col in text_cols:
            frame.loc[ind, col] = clean_texts(str(frame.loc[ind, col]))
    if test==False:
        tokenizer = Tokenizer(oov_token='OOV')
        for col in text_cols:
            tokenizer.fit_on_texts(frame[col])
    else:
        tokenizer=tokenizer
    renamed_cols = []
    for col in text_cols:
        renamed_cols.append(col+'_tokenized')
        frame[col+'_tokenized'] = tokenizer.texts_to_sequences(frame[col])
        frame.drop([col], inplace=True, axis=1)
    return [frame, renamed_cols, tokenizer]


def model_6_lstm(X_train_padded, tokenizer, maxlen=750):
    
    initializer = RandomUniform(seed=69)
    #j = 0
    embs = []
    inputs = []
    for each in X_train_padded:
        input_layer = Input(shape=(maxlen, ))
        emb_layer = Embedding(len(tokenizer.word_index)+1, output_dim=128)(input_layer)
        embs.append(emb_layer)
        inputs.append(input_layer)
    concat = Concatenate(axis=1)(embs)
    lstm_layer_1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(concat)
    lstm_layer_2 = Bidirectional(LSTM(128, dropout=0.1))(lstm_layer_1)
    norm_layer = BatchNormalization()(lstm_layer_2)
    dense_layer = Dense(256, activation='relu')(norm_layer)
    
    droput = Dropout(0.2)(dense_layer)
    dense_1 = Dense(128, activation='relu')(droput)
    dense_2 = Dense(64, activation='relu')(dense_1)
    output = Dense(30)(dense_2)

    model = Model(inputs=inputs, outputs=[output])
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    #print(model.summary())

    return model    

def spears_score(y_test, predictions):
    rhos = []
    for tcol, pcol in zip(np.transpose(y_test), np.transpose(predictions)):
        rhos.append(spearmanr(tcol, pcol).correlation)
    return np.nanmean(rhos)


def main():
    
    train_data = pd.read_csv('../input/google-quest-challenge/train.csv')
    test_data = pd.read_csv('../input/google-quest-challenge/test.csv')
    train_data = prepare_data(train_data)
    test_data = prepare_data(test_data)
    train_cols, _ = get_vars_and_targets(train_data, test_data)
    submission_data = pd.read_csv('../input/google-quest-challenge/sample_submission.csv', encoding='utf-8')
    target_cols = list(submission_data.columns[1:].values)
    
    X_train, X_test, y_train, y_test = train_test_split(train_data.loc[:, train_cols], train_data.loc[:, target_cols], test_size=0.25)
    X_train, text_cols, tokenizer = transform_texts(X_train.drop('qa_id', axis=1), test=False)
    X_test, text_cols, _ = transform_texts(X_test.drop('qa_id', axis=1), tokenizer, test=True)
    print(X_train.columns, X_test.columns)
    #scaler = MinMaxScaler()
    #scaler.fit(y_train.values)
    #y_train, y_test = scaler.transform(y_train), scaler.transform(y_test.values)
    X_train_features = []
    X_test_features = []
    for col in text_cols:
        X_train_features.append('X_train_'+(('_').join(col.split('_')[:-1])))
        X_test_features.append('X_test_'+(('_').join(col.split('_')[:-1])))
    X_train_padded = []
    X_test_padded = []
    i = 0
    for each in X_train_features:
        X_train_padded.append(pad_sequences(X_train[text_cols[i]], maxlen=750, padding='pre'))
        X_test_padded.append(pad_sequences(X_test[text_cols[i]], maxlen=750, padding='pre'))
        i += 1
    model = model_6_lstm(X_train_padded, tokenizer)
    model.fit(X_train_padded, y_train.values, batch_size=16, epochs=2, validation_split=0.2)
    print(model.evaluate(X_test_padded, y_test.values))
    print('Spearmen score:', spears_score(y_test.values, model.predict(X_test_padded)))

    test_data, text_cols, _ = transform_texts(test_data.drop(['qa_id'], axis=1), tokenizer=tokenizer, test=True)
    test_features = []
    for col in text_cols:
        test_features.append(col)
        test_padded = []
    j = 0
    for each in test_features:
        test_padded.append(pad_sequences(test_data[each], maxlen=750, padding='pre'))
        j += 1
        
    predictions = model.predict(test_padded)
    print(predictions)
    prediction = np.where(predictions<1, predictions, 0.99)
        
    submission_data[target_cols] = np.absolute(predictions)

    print(submission_data.head())
    submission_data.to_csv('submission.csv', index=False)
    
    

    
main()


# In[ ]:




