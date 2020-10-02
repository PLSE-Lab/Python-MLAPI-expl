#!/usr/bin/env python
# coding: utf-8

# # Quoras Question Pairs Modeling Notebook
# 
# This notebook try to predict if some pair of Quoras questions are duplicated or not.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from zipfile import ZipFile

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# First, lets get the train dataset.

# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_train.head()


# In[ ]:


texts = df_train[['question1','question2']]
labels = df_train['is_duplicate']

del df_train


# Now, lets build our model. First we need tokenize the questions to create a word index, then we use it with the Glove model as our embedding layer, that transforms the input vector with our words index to a dense vector that represents our words sequence. We need to load the pre-trained GloVe model in order to use it as our embedding model.

# In[ ]:


# Params
MAX_NB_WORDS = 100000
MAX_SEQUENCE_LENGTH = 40
VALIDATION_SPLIT = 0.1
EMBEDDING_DIM = 128


# Prepare the questions.

# In[ ]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

tk = Tokenizer(num_words=MAX_NB_WORDS)

tk.fit_on_texts(list(texts.question1.values.astype(str)) + list(texts.question2.values.astype(str)))
x1 = tk.texts_to_sequences(texts.question1.values.astype(str))
x1 = pad_sequences(x1, maxlen=MAX_SEQUENCE_LENGTH)

x2 = tk.texts_to_sequences(texts.question2.values.astype(str))
x2 = pad_sequences(x2, maxlen=MAX_SEQUENCE_LENGTH)

word_index = tk.word_index

#labels = to_categorical(labels)

print('Shape of data tensor:', x1.shape, x2.shape)
print('Shape of label tensor:', labels.shape)


# In[ ]:


from keras.layers import Dense, Dropout, Lambda, TimeDistributed, PReLU, Merge, Activation, Embedding
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras import backend as K

encoder_1 = Sequential()
encoder_1.add(Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH))

encoder_1.add(TimeDistributed(Dense(EMBEDDING_DIM, activation='relu')))
encoder_1.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(EMBEDDING_DIM,)))

encoder_2 = Sequential()
encoder_2.add(Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH))

encoder_2.add(TimeDistributed(Dense(EMBEDDING_DIM, activation='relu')))
encoder_2.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(EMBEDDING_DIM,)))

model = Sequential()
model.add(Merge([encoder_1, encoder_2], mode='concat'))
model.add(BatchNormalization())

model.add(Dense(EMBEDDING_DIM))
model.add(PReLU())
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(EMBEDDING_DIM))
model.add(PReLU())
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpoint = ModelCheckpoint('weights.h5', monitor='val_acc', save_best_only=True, verbose=2)

model.fit([x1, x2], y=labels, batch_size=384, nb_epoch=1,
                 verbose=1, validation_split=0.1, shuffle=True, callbacks=[checkpoint])


# As kaggle have time limit for running kernels, this models trains just one epoch and is pretty small. A bigger/depper model with proper training time will perform better.
