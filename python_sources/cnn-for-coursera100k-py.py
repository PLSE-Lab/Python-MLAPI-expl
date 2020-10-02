import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D, Conv1D, MaxPool1D, Embedding, Dropout
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split


##### For some reason kaggle makes error for reading my dataset :(  but on my own pc i reached accuracy of around 95%

dataset = pd.read_csv('../input/reviews.csv')
dataset.head()

X = dataset.iloc[:, 1]
Y = dataset.iloc[:,2]

print(len(X), len(Y))
print(set(Y))


Y = Y.map(lambda y : 2 if int(y)> 3 else 1 if int(y)==3 else 0)
Y = np_utils.to_categorical(Y)
print(Y[0])

print(max(len(s) for s in X))
print(min(len(s)for s in X))
sorted_X = sorted(len(s) for s in X)
print(sorted_X[len(sorted_X) // 2])


tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)

word2index = tokenizer.word_index
print(len(word2index))

sequences = pad_sequences(sequences, maxlen=90)
print(sequences.shape)

X_train, X_test, Y_train, Y_test = train_test_split(sequences, Y, test_size=0.1, random_state=0)

max_len = 90
vector_length = 100
input_dim = len(word2index)
batch_size = 32
epochs = 15

model = Sequential()
model.add(Embedding(input_dim=input_dim, output_dim=vector_length, input_length=max_len))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPool1D(3))
model.add(Dropout(0.2))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPool1D(3))
model.add(Dropout(0.2))
model.add(Conv1D(64, 3, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
save_best = ModelCheckpoint('coursera.hdf', save_best_only=True, monitor='val_loss', mode='min')

result = model.fit(X_train, Y_train, batch_size = batch_size, epochs=epochs, validation_split=0.1, verbose=1, callbacks=[save_best])

model.load_weights(filepath='coursera.hdf')
eval_ = model.evaluate(X_test, Y_test)
print(eval_[0], eval_[1]) # loss / accuracy

def plot_model(result):
    acc = result.history['acc']
    val_acc = result.history['val_acc']
    loss = result.history['loss']
    val_loss = result.history['val_loss']
    x = range(1, len(acc)+1)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label= 'Validation acc')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='validation loss')
    plt.legend()
    
plot_model(result)


# Test Loss : 0.12343048582793707

# Test Accuracy : 0.9523765214512981

