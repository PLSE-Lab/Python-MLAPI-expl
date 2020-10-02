#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

import tensorflow as tf
from keras.utils.data_utils import get_file
import tarfile

seed = 42
np.random.seed(seed)


# In[ ]:


plot_genre=pd.read_csv('../input/plots.txt', sep='\t', lineterminator='\n')


# In[ ]:


plot_genre.head()


# # WORD level, no encoder decoder

# In[ ]:


print(plot_genre.shape)
plot_genre.head()


# In[ ]:


plot_genre.dropna(inplace=True)


# In[ ]:


plot1="adventure|science fiction|detective" # select 3 genres to generate 
plots=plot_genre[plot_genre['genres'].str.contains(plot1)]['plot']
print(plots.shape)


# In[ ]:


mask = (plots.str.len() <= 1000) # drop plots that have more than 1000 chars
not_huge_plots = plots.loc[mask]
(not_huge_plots).shape


# In[ ]:


strings=not_huge_plots.values.T.tolist()
text1 = ''.join(str(e) for e in strings)


# In[ ]:


print ('Length of text: {} characters'.format(len(text1)))


# In[ ]:


#modified from https://github.com/enriqueav/lstm_lyrics/blob/master/lstm_train_embedding.py
from __future__ import print_function
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional, Embedding
import numpy as np
import random
import sys
import io
import os


# In[ ]:


# Parameters: change to experiment different configurations
SEQUENCE_LEN = 30
MIN_WORD_FREQUENCY = 10
STEP = 1
BATCH_SIZE = 1024


def shuffle_and_split_training_set(sentences_original, next_original, percentage_test=2):
    # shuffle at unison
    print('Shuffling sentences')

    tmp_sentences = []
    tmp_next_word = []
    
    for i in np.random.RandomState(seed=42).permutation(len(sentences_original)):
        tmp_sentences.append(sentences_original[i])
        tmp_next_word.append(next_original[i])

    cut_index = int(len(sentences_original) * (1.-(percentage_test/100.)))
    x_train, x_test = tmp_sentences[:cut_index], tmp_sentences[cut_index:]
    y_train, y_test = tmp_next_word[:cut_index], tmp_next_word[cut_index:]

    print("Size of training set = %d" % len(x_train))
    print("Size of test set = %d" % len(y_test))
    return (x_train, y_train), (x_test, y_test)


# Data generator for fit and evaluate
def generator(sentence_list, next_word_list, batch_size):
    index = 0
    while True:
        x = np.zeros((batch_size, SEQUENCE_LEN), dtype=np.int32)
        y = np.zeros((batch_size), dtype=np.int32)
        for i in range(batch_size):
            for t, w in enumerate(sentence_list[index % len(sentence_list)]):
                x[i, t] = word_indices[w]
            y[i] = word_indices[next_word_list[index % len(sentence_list)]]
            index = index + 1
        yield x, y


def get_model(dropout=0.2):
    print('Build model...')
    model = Sequential()
    model.add(Embedding(input_dim=len(words), output_dim=1024))
    model.add(Bidirectional(LSTM(256, return_sequences=False)))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(len(words)))
    model.add(Activation('softmax'))
    return model


# Functions from keras-team/keras/blob/master/examples/lstm_text_generation.py
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# In[ ]:


#os.listdir('../working/checkpoints/')


# In[ ]:


corpus = text1
#examples = '../working/examples.txt'

if not os.path.isdir('./checkpoints/'):
    os.makedirs('./checkpoints/')
file = open("../working/checkpoints/examples", "w") 
file.close() 

text = text1.replace('\n', ' \n ')
print('Corpus length in characters:', len(text))


# In[ ]:


text_in_words = [w for w in text.split(' ') if w.strip() != '' or w == '\n']
print('Corpus length in words:', len(text_in_words))


# In[ ]:


word_freq = {}
for word in text_in_words:
    word_freq[word] = word_freq.get(word, 0) + 1
words = set(text_in_words)
print('Unique words:', len(words))


# In[ ]:


word_indices = dict((c, i) for i, c in enumerate(words))
indices_word = dict((i, c) for i, c in enumerate(words))


# In[ ]:


# cut the text in semi-redundant sequences of SEQUENCE_LEN words
sentences = []
next_words = []
ignored = 0
for i in range(0, len(text_in_words) - SEQUENCE_LEN, STEP):
        sentences.append(text_in_words[i: i + SEQUENCE_LEN])
        next_words.append(text_in_words[i + SEQUENCE_LEN])
print('All sequences:', len(sentences))


# In[ ]:


# x, y, x_test, y_test
(sentences, next_words), (sentences_test, next_words_test) = shuffle_and_split_training_set(sentences, next_words)

model = get_model()
model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

file_path = "../working/checkpoints/examples"
model.summary()


# In[ ]:


checkpoint = ModelCheckpoint(file_path, monitor='val_acc', save_best_only=True, save_weights_only=True)
#print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
#early_stopping = EarlyStopping(monitor='val_acc', patience=50)
callbacks_list = [checkpoint]


# In[ ]:


h1=model.fit_generator(generator(sentences, next_words, BATCH_SIZE),
                    steps_per_epoch=int(len(sentences)/BATCH_SIZE) + 1,
                    epochs=50,
                    callbacks=callbacks_list,
                    validation_data=generator(sentences_test, next_words_test, BATCH_SIZE),
                    validation_steps=int(len(sentences_test)/BATCH_SIZE) + 1)


# In[ ]:


# generate text
for diversity in [0.2, 0.5, 1.0, 1.2]:
    seed_index = np.random.randint(len(sentences+sentences_test))
    seed = (sentences+sentences_test)[seed_index]
    sentence=seed
    print('----- diversity:', diversity)

    print('----- Generating with seed:\n"' + ' '.join(sentence) + '"\n')
    #print(' '.join(sentence))

    for i in range(200):
        x_pred = np.zeros((1, SEQUENCE_LEN))
        for t, word in enumerate(sentence):
            x_pred[0, t] = word_indices[word]

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_word = indices_word[next_index]

        sentence = sentence[1:]
        sentence.append(next_word)


        sys.stdout.write(" "+next_word)
        sys.stdout.flush()
    print()
    print()


# # to be continued

# In[ ]:


model.save_weights(file_path) # save weights for later use
#os.listdir('../working/checkpoints/')
statinfo = os.stat('../working/checkpoints/examples')
size_w=statinfo.st_size/1024/1024
print(size_w)


# In[ ]:




