import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
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
from keras.layers import LSTM, Input, Dot, Softmax, Multiply, Concatenate, Subtract, Dense, Lambda, Embedding, Dropout
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




bilstm1 = Bidirectional(LSTM(300, return_sequences=True))
bilstm2 = Bidirectional(LSTM(300, return_sequences=True))

claimInput = Input(shape=(maxLength,), dtype='float32')
evidenceInput = Input(shape=(maxLength,), dtype='float32')

x1 = embed(claimInput)
x2 = embed(evidenceInput)

aBar = bilstm1(x1)
bBar = bilstm1(x2)

# Soft-alignment layer computes attention weights as similarity between claim and evidence 
e = Dot(axes=2)([aBar, bBar]) # EQTN 11

# Local inference then collected over sequences
# The attention weight is used to derive the relevance between the claim and evidence 
# For the hidden state of a word identify the relevant semantics in the evidence

# EQTN 12
e1 = Softmax(axis=2)(e)
e1 = Lambda(K.expand_dims, arguments={'axis': 3})(e1)
aTilde = Lambda(K.expand_dims, arguments={'axis': 1})(bBar)
aTilde = Multiply()([e1, aTilde])
aTilde = Lambda(K.sum, arguments={'axis': 2})(aTilde)

# EQTN 13
e2 = Softmax(axis=1)(e)
e2 = Lambda(K.expand_dims, arguments={'axis': 3})(e2)
bTilde = Lambda(K.expand_dims, arguments={'axis': 2})(aBar)
bTilde = Multiply()([e2, bTilde])
bTilde = Lambda(K.sum, arguments={'axis': 1})(bTilde)

# Enchancement of the local inference information collected
# Such operations can help sharpen the local inference information and capture inference relationships such as contradiction

# EQTN 14
mA = Concatenate()([aBar, aTilde, Subtract()([aBar, aTilde]), Multiply()([aBar, aTilde])])

# EQTN 15
mB = Concatenate()([bBar, bTilde, Subtract()([bBar, bTilde]), Multiply()([bBar, bTilde])])

# Composition layer to compose the enhanced local inference information m1 and m2 above
yA = bilstm2(mA)
yB = bilstm2(mB)

# Compute average and max pooling and concat these vectors to form the final fixed length vector (V)
# This leads to better results than summation
# EQTN 18
maxA = Lambda(K.max, arguments={'axis' : 1})(yA)
avA = Lambda(K.mean, arguments={'axis' : 1})(yA)

# EQTN 19
maxB = Lambda(K.max, arguments={'axis' : 1})(yB)
avB = Lambda(K.mean, arguments={'axis' : 1})(yB)

y = Concatenate()([avA, maxA, avB, maxB])

# Finally pass (V) through a MLP - MLP has a hidden layer with tanh activation and softmax output layer
y = Dense(1024, activation='tanh')(y)
y = Dropout(0.5)(y)
y = Dense(2, activation='sigmoid')(y)

model = Model(inputs=[claimInput, evidenceInput], outputs=y)
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit([training[0], training[1]], training[2], batch_size=batchSize, epochs=15, verbose=1)

# loss, accuracy = model.evaluate([training[0], training[1]], training[2], verbose=False)
# print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate([test[0], test[1]], test[2], verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))


y_preds = model.predict([test[0], test[1]])
y_preds = np.argmax(y_preds, axis=1)

y_test = np.argmax(test[2], axis=1)

print('Precision', precision_score(y_test, y_preds))
print('Recall', recall_score(y_test, y_preds))
print('F1 score', f1_score(y_test, y_preds))

