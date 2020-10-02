# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

### DISCLAIMER : This is my first attempt at using Keras for an NLP task. The code I have here has been sourced from a Keras blog - 
###              https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html - with a few modifications of my own.
###
###              I would like to thank Francois Chollet for a simple and neat tutorial. 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np

import os
import pickle

import sklearn.metrics as skm
import keras 
from keras import utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras import backend as K
from keras.models import Sequential, Model
from keras import optimizers
from keras import metrics
from keras.wrappers.scikit_learn import KerasRegressor
from keras import regularizers
from keras.models import load_model
from keras.layers.normalization import BatchNormalization


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))

# Loading Train and Test Data
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

# Prepare training data

texts = train_data.question_text.values.tolist()  # list of text samples
texts = [t.strip().lower() for t in texts]

labels = train_data.target.values.tolist()  # list of label ids

# Defining preprocessing variables
MAX_NB_WORDS = 25000
MAX_SEQUENCE_LENGTH = 70
VALIDATION_SPLIT = 0.2

# Tokenizatiob process
tokenizer = Tokenizer(num_words = MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Word index dictionary
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# Padding sequences for consistency
data = pad_sequences(sequences, maxlen = MAX_SEQUENCE_LENGTH)

# Preparing labels
labels = keras.utils.to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

# Split data into training and validation set
X_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
X_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

# Preparing the embedding matrix
embeddings_index = {}
f = open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt', encoding="utf-8")

for line in f:
    values = line.split()
    word = values[0]
    try:
        coefs = np.asarray(values[1:], dtype='float32')
    except ValueError as ve:
        continue
    
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

EMBEDDING_DIM = 300

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# Defining a pre-trained embedding layer in Keras
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


# Model definition
def simple_CNN():
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(15)(x)  # global max pooling
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(2, activation='softmax')(x)
    
    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    return model
    
def simple_LSTM():
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    layer = LSTM(64)(embedded_sequences)
    layer = Dense(256, name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(2, name = 'out_layer')(layer)
    preds = Activation('sigmoid')(layer)
    
    model = Model(inputs = sequence_input, outputs = preds)
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])
    
    return model
    
def simple_GRU():
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    layer = GRU(64)(embedded_sequences)
    layer = Dense(256, name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(2, name = 'out_layer')(layer)
    preds = Activation('sigmoid')(layer)
    
    model = Model(inputs = sequence_input, outputs = preds)
    
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])
    
    return model
    
def bi_GRU():
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    
    layer_fwd = GRU(64)(embedded_sequences)
    layer_bwd = GRU(64, go_backwards = True)(embedded_sequences)
    
    layer = concatenate([layer_fwd, layer_bwd])
    
    layer = Dense(256, name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(64, name='FC2')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(2, name = 'out_layer')(layer)
    preds = Activation('sigmoid')(layer)
    
    model = Model(inputs = sequence_input, outputs = preds)
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])
    
    return model
    
def hybrid_cnn_gru():
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    
    x = Conv1D(256, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    
    layer = GRU(64)(x)
    
    layer = Dense(256, name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(2, name = 'out_layer')(layer)
    preds = Activation('sigmoid')(layer)
    
    model = Model(inputs = sequence_input, outputs = preds)
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])
    
    return model

class_0_weight = (np.sum(y_train == 1) * 1.) / np.sum(y_train == 0)
class_1_weight = (np.sum(y_train == 0) * 1.) / np.sum(y_train == 1)

APPLY_ATTENTION_BEFORE_LSTM = True
model = simple_GRU()

# Model training
model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=4, batch_size=128, 
          class_weight = {0: class_0_weight, 1: class_1_weight})


def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = skm.roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 


# Function to prepare test data
def prepare_test_data(test_data, tokenizer, MAX_SEQUENCE_LENGTH):
    
    test_texts = test_data.question_text.values.tolist()  # list of text samples
    test_texts = [t.strip().lower() for t in test_texts]
    
    test_sequences = tokenizer.texts_to_sequences(test_texts)
    test_data_sequence = pad_sequences(test_sequences, maxlen = MAX_SEQUENCE_LENGTH)
    
    return test_data_sequence

test_data_texts = prepare_test_data(test_data, tokenizer, MAX_SEQUENCE_LENGTH)


#model_train_preds = model.predict(X_train)[:, 1]
#model_test_preds = model.predict(X_val)[:, 1]

# Find optimal probability threshold
#threshold = Find_Optimal_Cutoff(y_train[:, 1], conv_net_train_preds)
#print ('Optimal cutoff - ', threshold)

# Predict on test data
kaggle_submission_pred_probs = model.predict(test_data_texts)[:, 1]
kaggle_submission_preds = np.array([1 if i > 0.51 else 0 for i in kaggle_submission_pred_probs])
test_data['prediction'] = kaggle_submission_preds

# Saving test data predictions
test_data[['qid', 'prediction']].to_csv('submission.csv', index = False)

# Any results you write to the current directory are saved as output.