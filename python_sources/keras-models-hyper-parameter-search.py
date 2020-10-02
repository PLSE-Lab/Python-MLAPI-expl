import numpy as np
import pandas as pd
import sys
import glob
import os
import keras
import itertools

from collections import Counter
from gensim.models import word2vec
from os.path import join, exists, split

from sklearn import preprocessing
from sklearn.metrics import log_loss, classification_report
from sklearn.model_selection import KFold

from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, Embedding, Bidirectional, LSTM, GRU
from keras.models import Model
from keras.layers import Flatten
import tensorflow as tf
from keras import backend as K

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_all = pd.concat([df_train, df_test])
df_all.head()
print(len(df_train), len(df_test), len(df_all))

import re
def clean_str(string):
    string = re.sub(r"[^a-zA-Z0-9¿¡¬√ƒ≈«»… ÀÃÕŒœ“”‘’÷Ÿ⁄€‹›‡·‚„‰ÂÁËÈÍÎÏÌÓÔÚÛÙıˆ˘˙˚¸˝ˇ,!?\'\`\.\(\)]", " ", string)
    string = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " <IPADDRESS> ", string)
    string = re.sub(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}\b", " <EMAIL> ", string)
    string = re.sub(r"[+-]?\d+(?:\.\d+)?", " <NUMBER> ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

df_all['comment_text_clean'] = df_all['comment_text'].apply(clean_str)
df_all['comment_text_clean'] = df_all['comment_text_clean'].apply(lambda x : x.split(' '))

df_all['comment_size'] = df_all['comment_text_clean'].apply(len)

def pad_sentence(sentence, sequence_length, padding_word="<PAD/>"):
    if len(sentence) > sequence_length:
        return sentence[:sequence_length]
    else:
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        return new_sentence

    
MAX_SEQUENCE_LEN = 650

df_all['comment_text_clean'] = df_all['comment_text_clean'].apply(lambda x : pad_sentence(x, MAX_SEQUENCE_LEN))


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return (vocabulary, vocabulary_inv)

comments = df_all['comment_text_clean'].values

vocabulary, vocabulary_inv = build_vocab(comments)
print("Vocabulary Size: {:d}".format(len(vocabulary)))

def build_input_data(sentences, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    #y = np.array(labels)
    return x

X = build_input_data(comments, vocabulary)


def train_word2vec(sentence_matrix, vocabulary_inv,
                   num_features=300, min_word_count=1, context=10):
    """
    Trains, saves, loads Word2Vec model
    Returns initial weights for embedding layer.
   
    inputs:
    sentence_matrix # int matrix: num_sentences x max_sentence_len
    vocabulary_inv  # dict {str:int}
    num_features    # Word vector dimensionality                      
    min_word_count  # Minimum word count                        
    context         # Context window size 
    """
    model_dir = 'word2vec_models'
    model_name = "{:d}features_{:d}minwords_{:d}context".format(num_features, min_word_count, context)
    model_name = join(model_dir, model_name)
    if exists(model_name):
        embedding_model = word2vec.Word2Vec.load(model_name)
        print('Loading existing Word2Vec model \'%s\'' % split(model_name)[-1])
    else:
        # Set values for various parameters
        num_workers = 7       # Number of threads to run in parallel
        downsampling = 1e-3   # Downsample setting for frequent words
        
        # Initialize and train the model
        print("Training Word2Vec model...")
        sentences = [[vocabulary_inv[w] for w in s] for s in sentence_matrix]
        embedding_model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count = min_word_count,                             window = context, sample = downsampling)
        
        # If we don't plan to train the model any further, calling 
        # init_sims will make the model much more memory-efficient.
        embedding_model.init_sims(replace=True)
        
        # Saving the model for later use. You can load it later using Word2Vec.load()
        if not exists(model_dir):
            os.mkdir(model_dir)
        print('Saving Word2Vec model \'%s\'' % split(model_name)[-1])
        embedding_model.save(model_name)
    
    #  add unknown words
    embedding_weights = [np.array([embedding_model[w] if w in embedding_model                                                        else np.random.uniform(-0.25,0.25,embedding_model.vector_size)                                                        for w in vocabulary_inv])]
    return embedding_weights

idx_train = len(df_train)

df_train = df_all.iloc[:idx_train]
df_test = df_all.iloc[idx_train:]

X_train = np.array(X[:idx_train])
X_test = np.array(X[idx_train:])
y_train = df_train[['identity_hate','insult','obscene','severe_toxic','threat','toxic']].values


print(X_train.shape, y_train.shape)
num_classes = 6

def create_model_cnn(trainable_embedding=False, dropout=0.3, dense_units=128, embedding_weights=None, embedding_dim=100):
    sequence_input = Input(shape=(MAX_SEQUENCE_LEN,), dtype='int32')
    embedding_layer = Embedding(len(vocabulary), embedding_dim, input_length=MAX_SEQUENCE_LEN, 
                                weights=embedding_weights, trainable=trainable_embedding)

    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 1, activation='relu')(embedded_sequences)
    x = MaxPooling1D(2)(x)
    x = Conv1D(128, 2, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(128, 4, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Flatten()(x)
    x = Dense(dense_units, activation='relu')(x)
    x = Dropout(dropout)(x)
    preds = Dense(num_classes, activation='sigmoid')(x)
  
    model = Model(sequence_input, preds)  
    model.name = 'CNN'
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model

def create_model_lstm(trainable_embedding=False, dropout=0.4, units=64, embedding_weights=None, embedding_dim=100):
    sequence_input = Input(shape=(MAX_SEQUENCE_LEN,), dtype='int32')
    embedding_layer = Embedding(len(vocabulary), embedding_dim, input_length=MAX_SEQUENCE_LEN, 
                                weights=embedding_weights, trainable=trainable_embedding)

    embedded_sequences = embedding_layer(sequence_input)

    x = Bidirectional(LSTM(units))(embedded_sequences)
    x = Dropout(dropout)(x)
    preds = Dense(num_classes, activation='sigmoid')(x)
    model = Model(sequence_input, preds)  
    model.name = 'LSTM'
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model


def create_model_gru(trainable_embedding=False, dropout=0.4, units=64, embedding_weights=None, embedding_dim=100):
    sequence_input = Input(shape=(MAX_SEQUENCE_LEN,), dtype='int32')
    embedding_layer = Embedding(len(vocabulary), embedding_dim, input_length=MAX_SEQUENCE_LEN, 
                                weights=embedding_weights, trainable=trainable_embedding)

    embedded_sequences = embedding_layer(sequence_input)

    x = GRU(units)(embedded_sequences)
    x = Dropout(dropout)(x)
    preds = Dense(num_classes, activation='sigmoid')(x)
    model = Model(sequence_input, preds)  
    model.name = 'GRU'
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model


def save_file(name, file_type, values, ids):
    df_novo = ids
    df_novo.loc[:,'identity_hate'] = values[:,0]
    df_novo.loc[:,'insult'] = values[:,1]
    df_novo.loc[:,'obscene'] = values[:,2]
    df_novo.loc[:,'severe_toxic'] = values[:,3]
    df_novo.loc[:,'threat'] = values[:,4]
    df_novo.loc[:,'toxic'] = values[:,5]
    df_novo[['id','identity_hate','insult','obscene','severe_toxic','threat',
             'toxic']].to_csv('./submit/' + file_type + '_' + name + '.csv', float_format='%.8f', index=None)


kf = KFold(n_splits=3, shuffle=True, random_state=5)

embedding_dim = [50,100]
min_word_count = [1,2]
contexts = [8,10]
units = [64, 128, 256]
dropout = [0.4]
trainable = [True, False]
f_models = [create_model_cnn, create_model_gru, create_model_lstm]

for emb in embedding_dim:
    for min_word in min_word_count:
        for context in contexts:
            for m in f_models:
                for unit in units:
                    for drop in dropout:
                        for tr in trainable:
                            preds = np.zeros((y_train.shape[0],y_train.shape[1]))
                            
                            for train, valid in kf.split(X_train):
                                embedding_weights = train_word2vec(X, vocabulary_inv, emb, min_word, context)
                                model = m(tr, drop, unit, embedding_weights, emb)
                                model.fit(X_train[train], y_train[train], epochs=2, batch_size=64, 
                                          verbose=1, validation_data=(X_train[valid], y_train[valid]))
                                preds[valid] = model.predict(X_train[valid])
                            file_name = str(emb) + '_' + str(min_word) + '_' + str(context) + '_' + str(model.name)                                             + '_' + str(unit) + '_' + str(drop) + '_' + str(tr)
                            print(file_name)
                            save_file(file_name, 'train', preds, df_train[['id']])
                            preds_test = model.predict(X_test)
                            save_file(file_name, 'test', preds_test, df_test[['id']])


