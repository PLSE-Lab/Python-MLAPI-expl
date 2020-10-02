#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function
import collections
import os
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
import re
from sklearn.utils import shuffle


# In[ ]:


data = pd.read_csv('../input/neural-network/neural_network.csv')
abstracts = list(data['patent_abstract'])
print(abstracts)


# In[ ]:


# Data cleaning

def format_patent(patent):
    """Add spaces around punctuation and remove references to images/citations."""

    # Add spaces around punctuation
    patent = re.sub(r'(?<=[^\s0-9])(?=[.,;?])', r' ', patent)

    # Remove references to figures
    patent = re.sub(r'\((\d+)\)', r'', patent)

    # Remove double spaces
    patent = re.sub(r'\s\s', ' ', patent)
    return patent


formatted = []

for abstract in abstracts:
    formatted.append(format_patent(abstract))


# In[ ]:


tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower= True)
tokenizer.fit_on_texts(formatted)

# Create look-up dictionaries and reverse look-ups
word_idx = tokenizer.word_index
idx_word = tokenizer.index_word
num_words = len(word_idx) + 1
word_counts = tokenizer.word_counts

print(f"There are {num_words} unique words")

sequences = tokenizer.texts_to_sequences(formatted)

sequence_lengths = [len(sequence) for sequence in sequences]

over_idx = []
for i, l in enumerate(sequence_lengths):
    if l > 70:
        over_idx.append(i)

new_texts = []
new_sequences = []

for i in over_idx:
    new_texts.append(formatted[i])
    new_sequences.append(sequences[i])

training_sequences = []
labels = []

for sequence in new_sequences:
    for i in range(50, len(sequence)):
        extract = sequence[i-50:i+1]
        training_sequences.append(extract[:-1])
        labels.append(extract[-1])

print(f'There are {len(training_sequences)} training sequences')


# In[ ]:


feats = ' '.join(idx_word[i] for i in training_sequences[4])
answer = idx_word[labels[4]]

print('Features:', feats)
print('\nLabel: ', answer)


# In[ ]:


# Randomly shuffle features and labels
features, labels = shuffle(training_sequences, labels, random_state=0)

# Decide on number of samples for training
train_end = int(0.7 * len(labels))

train_features = np.array(features[:train_end])
valid_features = np.array(features[train_end:])

train_labels = labels[:train_end]
valid_labels = labels[train_end:]

# Convert to arrays
X_train, X_valid = np.array(train_features), np.array(valid_features)

# Using int8 for memory savings
y_train = np.zeros((len(train_labels), num_words), dtype=np.int8)
y_valid = np.zeros((len(valid_labels), num_words), dtype=np.int8)

# One hot encoding of labels
for example_index, word_index in enumerate(train_labels):
    y_train[example_index, word_index] = 1

for example_index, word_index in enumerate(valid_labels):
    y_valid[example_index, word_index] = 1

# Memory management
import gc
gc.enable()
del features, labels, train_features, valid_features, train_labels, valid_labels
gc.collect()

print(X_train.shape)
print(y_train.shape)


# In[ ]:


from keras.utils import get_file

# glove_vectors = get_file('glove.6B.zip',
# 'http://nlp.stanford.edu/data/glove.6B.zip')
# #os.system(f'unzip {glove_vectors}')

glove_vectors = '../input/glove6b100dtxt/glove.6B.100d.txt'
glove = np.loadtxt(glove_vectors, dtype='str', comments=None)
print(glove.shape)


# In[ ]:


vectors = glove[:, 1:].astype('float')
words = glove[:, 0]

del glove

vectors[100], words[100]


# In[ ]:


vectors.shape


# In[ ]:


word_lookup = {word: vector for word, vector in zip(words, vectors)}
embedding_matrix = np.zeros((num_words, vectors.shape[1]))
not_found = 0

for i, word in enumerate(word_idx.keys()):
    # Look up the word embedding
    vector = word_lookup.get(word, None)

    # Record in matrix
    if vector is not None:
        embedding_matrix[i + 1, :] = vector
    else:
        not_found += 1

print(f'There were {not_found} words without pre-trained embeddings.')


# In[ ]:


import gc
gc.enable()
del vectors
gc.collect()


# In[ ]:


# Normalize and convert nan to 0
embedding_matrix = embedding_matrix /     np.linalg.norm(embedding_matrix, axis=1).reshape((-1, 1))
embedding_matrix = np.nan_to_num(embedding_matrix)


# In[ ]:


def find_closest(query, embedding_matrix, word_idx, idx_word, n=10):
    """Find closest words to a query word in embeddings"""

    idx = word_idx.get(query, None)
    # Handle case where query is not in vocab
    if idx is None:
        print(f'{query} not found in vocab.')
        return
    else:
        vec = embedding_matrix[idx]
        # Handle case where word doesn't have an embedding
        if np.all(vec == 0):
            print(f'{query} has no pre-trained embedding.')
            return
        else:
            # Calculate distance between vector and all others
            dists = np.dot(embedding_matrix, vec)

            # Sort indexes in reverse order
            idxs = np.argsort(dists)[::-1][:n]
            sorted_dists = dists[idxs]
            closest = [idx_word[i] for i in idxs]

    print(f'Query: {query}\n')
    max_len = max([len(i) for i in closest])
    # Print out the word and cosine distances
    for word, dist in zip(closest, sorted_dists):
        print(f'Word: {word:15} Cosine Similarity: {round(dist, 4)}')


# In[ ]:


find_closest('the', embedding_matrix, word_idx, idx_word)


# In[ ]:


print(np.dot(embedding_matrix, embedding_matrix[2]))


# In[ ]:


from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Embedding, Masking, Bidirectional
from keras.optimizers import Adam

from keras.utils import plot_model


# In[ ]:


def make_word_level_model(num_words,
                          embedding_matrix,
                          lstm_cells=64,
                          trainable=False,
                          lstm_layers=1,
                          bi_direc=False):
    """Make a word level recurrent neural network with option for pretrained embeddings
       and varying numbers of LSTM cell layers."""

    model = Sequential()

    # Map words to an embedding
    if not trainable:
        model.add(
            Embedding(
                input_dim=num_words,
                output_dim=embedding_matrix.shape[1],
                weights=[embedding_matrix],
                trainable=False,
                mask_zero=True))
        model.add(Masking())
    else:
        model.add(
            Embedding(
                input_dim=num_words,
                output_dim=embedding_matrix.shape[1],
                weights=[embedding_matrix],
                trainable=True))

    # If want to add multiple LSTM layers
    if lstm_layers > 1:
        for i in range(lstm_layers - 1):
            model.add(
                LSTM(
                    lstm_cells,
                    return_sequences=True,
                    dropout=0.1,
                    recurrent_dropout=0.1))

    # Add final LSTM cell layer
    if bi_direc:
        model.add(
            Bidirectional(
                LSTM(
                    lstm_cells,
                    return_sequences=False,
                    dropout=0.1,
                    recurrent_dropout=0.1)))
    else:
        model.add(
            LSTM(
                lstm_cells,
                return_sequences=False,
                dropout=0.1,
                recurrent_dropout=0.1))
    model.add(Dense(128, activation='relu'))
    # Dropout for regularization
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(num_words, activation='softmax'))

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model


model = make_word_level_model(
    num_words,
    embedding_matrix=embedding_matrix,
    lstm_cells=64,
    trainable=False,
    lstm_layers=1)
model.summary()


# In[ ]:


from keras.callbacks import EarlyStopping, ModelCheckpoint

BATCH_SIZE = 2048


def make_callbacks(model_name, save= True):
    """Make list of callbacks for training"""
    callbacks = [EarlyStopping(monitor='val_loss', patience=5)]

    if save:
        callbacks.append(
            ModelCheckpoint(
                f'{model_name}.h5',
                save_best_only=True,
                save_weights_only=False))
    return callbacks


callbacks = make_callbacks('pre-trained-rnn')


# In[ ]:


history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=BATCH_SIZE,
    verbose=1,
    callbacks=callbacks,
    validation_data=(X_valid, y_valid))


# In[ ]:


def load_and_evaluate(model_name, return_model=False):
    """Load in a trained model and evaluate with log loss and accuracy"""

    model = load_model(f'{model_name}.h5')
    r = model.evaluate(X_valid, y_valid, batch_size=2048, verbose=1)

    valid_crossentropy = r[0]
    valid_accuracy = r[1]

    print(f'Cross Entropy: {round(valid_crossentropy, 4)}')
    print(f'Accuracy: {round(100 * valid_accuracy, 2)}%')

    if return_model:
        return model


# In[ ]:


loaded_model = load_and_evaluate('pre-trained-rnn', return_model=True)


# In[ ]:


# Preparing data for training own embeddings 
tokenizer = Tokenizer(filters='!"%;[\\]^_`{|}~\t\n', lower= False)
tokenizer.fit_on_texts(formatted)

# Create look-up dictionaries and reverse look-ups
word_idx = tokenizer.word_index
idx_word = tokenizer.index_word
num_words = len(word_idx) + 1
word_counts = tokenizer.word_counts

print(f"There are {num_words} unique words")

sequences = tokenizer.texts_to_sequences(formatted)

sequence_lengths = [len(sequence) for sequence in sequences]

over_idx = []
for i, l in enumerate(sequence_lengths):
    if l > 70:
        over_idx.append(i)

new_texts = []
new_sequences = []

for i in over_idx:
    new_texts.append(formatted[i])
    new_sequences.append(sequences[i])

training_sequences = []
labels = []

for sequence in new_sequences:
    for i in range(50, len(sequence)):
        extract = sequence[i-50:i+1]
        training_sequences.append(extract[:-1])
        labels.append(extract[-1])

print(f'There are {len(training_sequences)} training sequences')


# In[ ]:


embedding_matrix = np.zeros((num_words, len(word_lookup['the'])))

not_found = 0

for i, word in enumerate(word_idx.keys()):
    # Look up the word embedding
    vector = word_lookup.get(word, None)

    # Record in matrix
    if vector is not None:
        embedding_matrix[i + 1, :] = vector
    else:
        not_found += 1

print(f'There were {not_found} words without pre-trained embeddings.')
embedding_matrix.shape


# In[ ]:



features, labels = shuffle(training_sequences, labels, random_state=0)

# Decide on number of samples for training
train_end = int(0.7 * len(labels))

train_features = np.array(features[:train_end])
valid_features = np.array(features[train_end:])

train_labels = labels[:train_end]
valid_labels = labels[train_end:]

# Convert to arrays
X_train, X_valid = np.array(train_features), np.array(valid_features)

# Using int8 for memory savings
y_train = np.zeros((len(train_labels), num_words), dtype=np.int8)
y_valid = np.zeros((len(valid_labels), num_words), dtype=np.int8)

# One hot encoding of labels
for example_index, word_index in enumerate(train_labels):
    y_train[example_index, word_index] = 1

for example_index, word_index in enumerate(valid_labels):
    y_valid[example_index, word_index] = 1

# Memory management
import gc
gc.enable()
del features, labels, train_features, valid_features, train_labels, valid_labels
gc.collect()

print(X_train.shape)
print(y_train.shape)


# In[ ]:


model = make_word_level_model(
    num_words,
    embedding_matrix,
    lstm_cells=64,
    trainable=True,
    lstm_layers=1)
model.summary()


# In[ ]:


model_name = 'train-embeddings-rnn'

callbacks = make_callbacks(model_name)


# In[ ]:


model.compile(
    optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    verbose=1,
    epochs= 100,
    callbacks=callbacks,
    validation_data=(X_valid, y_valid))

