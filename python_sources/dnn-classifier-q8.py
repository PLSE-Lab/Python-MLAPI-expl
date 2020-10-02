import numpy as np
import pandas as pd
import keras
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import merge, recurrent, Dense, Input, Dropout, TimeDistributed, concatenate, recurrent
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2
from keras.utils import np_utils
from keras.models import Sequential
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
import pickle 
import os
import zipfile
from collections import Counter

print(os.listdir("../input"))




training_f = open('../input/training-d/training_data.pickle', 'rb')
training = pickle.load(training_f)
training_f.close()
print('Loading')

testing_f = open('../input/testing/testing_data.pickle', 'rb')
test = pickle.load(testing_f)
testing_f.close()


tokenizer = Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(training[0] + training[1])


training = (training[0], training[1], np_utils.to_categorical(training[2], 2))
test = (test[0], test[1], np_utils.to_categorical(test[2], 2))


# Lowest index from the tokenizer is 1 - we need to include 0 in our vocab count
VOCAB = len(tokenizer.word_counts) + 1


batchSize = 512
maxLength = 30
dropout = 0.2 
activation = 'relu'


# Pad the sequences to a max length of maxLength
pad_sequence = lambda X: pad_sequences(tokenizer.texts_to_sequences(X), maxlen=maxLength)
# Apply sequence pad function to data
metaPadder = lambda data: (pad_sequence(data[0]), pad_sequence(data[1]), data[2])

training = metaPadder(training)
test = metaPadder(test)


GLOVE_PATH = '../input/precomputed-glove-weights/precomputed_glove.weights.npy' 
print('Loading GloVe')
embedding_matrix = np.load(GLOVE_PATH)

print('Total number of null word embeddings:')
print(np.sum(np.sum(embedding_matrix, axis=1) == 0))

embed = Embedding(VOCAB, 300, weights=[embedding_matrix], input_length=maxLength, trainable=False)


translate = TimeDistributed(Dense(300, activation=activation))

embSum = keras.layers.core.Lambda(lambda x: K.sum(x, axis=1), output_shape=(300, )) 

claim = Input(shape=(maxLength,), dtype='int32')
clm = embed(claim)
clm = translate(clm)
clm = embSum(clm)
clm = BatchNormalization()(clm)

evidence = Input(shape=(maxLength,), dtype='int32')
evdnce = embed(evidence)
evdnce = translate(evdnce)
evdnce = embSum(evdnce)
evdnce = BatchNormalization()(evdnce)

joint = concatenate([clm, evdnce])

joint = Dropout(dropout)(joint)
for i in range(4):
  joint = Dense(2 * 300, activation=activation)(joint)
  joint = Dropout(dropout)(joint)
  joint = BatchNormalization()(joint)
  
pred = Dense(2, activation='sigmoid')(joint)

model = Model(input=[claim, evidence], output=pred)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

print('Training')

model.fit([training[0], training[1]], training[2], batch_size=batchSize, nb_epoch=10, verbose=1)

loss, acc = model.evaluate([test[0], test[1]], test[2], batch_size=batchSize)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

y_preds = model.predict([test[0], test[1]])
y_preds = np.argmax(y_preds, axis=1)

y_test = np.argmax(test[2], axis=1)

print('Precision', precision_score(y_test, y_preds))
print('Recall', recall_score(y_test, y_preds))
print('F1 score', f1_score(y_test, y_preds))
