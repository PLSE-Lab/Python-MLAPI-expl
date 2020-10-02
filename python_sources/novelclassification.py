# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("."))


from keras.layers import Dense, Input, Dropout, MaxPooling1D, Conv1D, Activation
from keras.layers import LSTM, BatchNormalization

import numpy as np
from keras.models import Sequential

from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import string
from keras.layers import Embedding


MAX_LEN=453
NOVELS_CATEGORY=12
sentences=  open('../input/xtrain_obfuscated.txt').readlines()
test_sentences=open('../input/xtest_obfuscated.txt').readlines()
novels= open('../input/ytrain.txt').readlines()
 
TEST_SIZE=len(test_sentences)


def initialize_alpha_numeric(length):
    alpha_to_num = dict(zip(string.ascii_lowercase, range(1, length)))
    return alpha_to_num

def get_train_data(sentences,novels):
    total_size = len(sentences)
    x_train = np.zeros((total_size, MAX_LEN), dtype=np.int)
    for i in range(len(sentences)):
        x = np.zeros(MAX_LEN, dtype=np.int)
        for j in range(MAX_LEN):
            if j < len(sentences[i]) - 1:
                x[j] = initialize_alpha_numeric(27)[sentences[i][j]]
            else:
                x[j] = 0
        x_train[i] = np.copy(x)
    y_train = np.zeros((total_size, NOVELS_CATEGORY), dtype=np.int)

    for i in range(len(novels)):
        y_train[i][int(novels[i])] = 1
    return x_train,y_train

def get_test_data(test_sentences):
    total_size = len(test_sentences)
    x_test = np.zeros((total_size, MAX_LEN), dtype=np.int)
    for i in range(len(test_sentences)):
        x = np.zeros(MAX_LEN, dtype=np.int)
        for j in range(MAX_LEN):
            if j < len(test_sentences[i]) - 1:
                x[j] = initialize_alpha_numeric(27)[test_sentences[i][j]]
            else:
                x[j] = 0
        x_test[i] = np.copy(x)
    return  x_test



def train(epochs=10,batch_size=500):
    max_features=27 # unique characters in overall sentences in novel book
    embedding_size=13 # size of novel labels
    maxlen=453 # max length of sentence that can occur in
    model = Sequential()
    model.add(Embedding(max_features, embedding_size, input_length=maxlen))
    model.add(Dropout(0.20))

    model.add(BatchNormalization())
    model.add(Conv1D(filters=64,
                     kernel_size=5,
                     padding='valid',
                     activation='relu',
                     strides=1))

    model.add(MaxPooling1D(pool_size=4))

    model.add(LSTM(100))
    model.add(BatchNormalization())
    model.add(Dense(500))
    model.add(BatchNormalization())
    model.add(Dense(12))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    filepath="weights.best.hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    print('Training the model')
    history = model.fit(x_train, y_train, validation_split=0.1,
                        epochs=epochs, batch_size=batch_size,callbacks=callbacks_list)

    model_json = model.to_json()
    '''Serializing or saving the model object with weights'''
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

def predict():
    from keras.models import model_from_json
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    print("Model Loaded------")
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("weights.best.hdf5")
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    pred = loaded_model.predict(x_test)
    y_test = np.zeros(TEST_SIZE, dtype=np.int)
    for i in range(x_test.shape[0]):
        y_test[i] = np.argmax(pred[i])
    np.savetxt('ytest.txt', [y_test],fmt='%i', delimiter='\n')
    print("Completed")

x_train,y_train=get_train_data(sentences,novels)
x_test=get_test_data(test_sentences)
train()
predict()

# Any results you write to the current directory are saved as output.