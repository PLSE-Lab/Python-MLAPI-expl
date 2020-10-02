'''
CNN Binary classifer using GloVe embeddings 
'''
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.utils import np_utils
import random
from collections import OrderedDict
import numpy as np
import re
import pickle
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
import os
import zipfile
from collections import Counter
from keras.preprocessing.text import Tokenizer
print(os.listdir("../input"))


training_f = open('../input/training-d/training_data.pickle', 'rb')
training = pickle.load(training_f)
training_f.close()
print('Loading')

testing_f = open('../input/testing/testing_data.pickle', 'rb')
test = pickle.load(testing_f)
testing_f.close()

# ------------
# unique_items, counts = np.unique(training[2], return_counts=True)
# print(unique_items, counts)

# POSSAMPLES = 0
# NEGSAMPLES = 0

# newTraining1 = []
# newTraining2 = []
# newTrainingLbl = []

# for idx in range(len(training[2])):
#     if POSSAMPLES != 29775 and training[2][idx] == 1:
#         newTraining1.append(training[0][idx])
#         newTraining2.append(training[1][idx])
#         newTrainingLbl.append(training[2][idx])
#         POSSAMPLES += 1
    
#     if NEGSAMPLES != 29775 and training[2][idx] == 0:
#         newTraining1.append(training[0][idx])
#         newTraining2.append(training[1][idx])
#         newTrainingLbl.append(training[2][idx])
#         NEGSAMPLES += 1
        
# training = (newTraining1, newTraining2, newTrainingLbl)

# --------------
embedding_dim = 300
filter_sizes = [3, 4, 5]
num_filters = 128
std_drop = 0.6

epochs = 10
batch_size = 512
# vocabulary_size = len(vocab_dict)+1
max_length = 500


tokenizer = Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(training[0] + training[1])

zippedTrSamples = list(zip(training[0], training[1]))
trSamples = list(map(lambda x: x[0] + ' ' + x[1], zippedTrSamples))

zippedTeSamples = list(zip(test[0], test[1]))
teSamples = list(map(lambda x: x[0] + ' ' + x[1], zippedTeSamples))

training = (trSamples, np_utils.to_categorical(training[2], 2))
test = (teSamples, np_utils.to_categorical(test[2], 2))


# Lowest index from the tokenizer is 1 - we need to include 0 in our vocab count
VOCAB = len(tokenizer.word_counts) + 1

# Pad the sequences to a max length of MAX_LEN
pad_sequence = lambda X: pad_sequences(tokenizer.texts_to_sequences(X), maxlen=max_length)
# Apply sequence pad function to data
metaPadder = lambda data: (pad_sequence(data[0]), data[1])

training = metaPadder(training)
test = metaPadder(test)


GLOVE_PATH = '../input/precomputed-glove-weights/precomputed_glove.weights.npy' 
print('Loading GloVe')
embedding_matrix = np.load(GLOVE_PATH)

print('Total number of null word embeddings:')
print(np.sum(np.sum(embedding_matrix, axis=1) == 0))

embed = Embedding(VOCAB, embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False)


print("Creating Model...")
inputs = Input(shape=(max_length,), dtype='int32')
# embedding = Embedding(input_dim=VOCAB, output_dim=embedding_dim, input_length=max_length)(inputs)
embedding = embed(inputs)
reshape = Reshape((max_length, embedding_dim, 1))(embedding)

# Kernel size specifies the size of the 2-D conv window
# looking at 3 words at a time in the 1st layer, 4 in the 2nd ...
# set padding to valid to ensure no padding
conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal',
            activation='relu')(reshape)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal',
            activation='relu')(reshape)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal',
            activation='relu')(reshape)

# Pool size is the downscaling factor
maxpool_0 = MaxPool2D(pool_size=(max_length-filter_sizes[0]+1, 1), strides=(2,2), padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(max_length-filter_sizes[1]+1, 1), strides=(2,2), padding='valid')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(max_length-filter_sizes[2]+1, 1), strides=(2,2), padding='valid')(conv_2)

concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = Flatten()(concatenated_tensor)
dropout = Dropout(std_drop)(flatten)
output = Dense(units=2, activation='softmax')(dropout)

model = Model(inputs=inputs, outputs=output)

# checkpoint = ModelCheckpoint('model_flag_0.hdf5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
# adam = Adam(lr=2e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#print(model.summary())
print("Training Model...")
# model.fit(X_train_onehot, y_train_distribution, batch_size=batch_size, epochs=epochs, verbose=0, callbacks=[checkpoint],
#      validation_data=(X_dev_onehot, y_dev_distribution))

# model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
model.fit(training[0], training[1], batch_size=batch_size, epochs=epochs, verbose=1)

loss, accuracy = model.evaluate(test[0], test[1], verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))


y_preds = model.predict(test[0])
y_preds = np.argmax(y_preds, axis=1)

y_test = np.argmax(test[1], axis=1)
print(y_preds[:5])
print(y_test[:5])

print('Precision', precision_score(y_test, y_preds))
print('Recall', recall_score(y_test, y_preds))
print('F1 score', f1_score(y_test, y_preds))
