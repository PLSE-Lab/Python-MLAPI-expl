#!/usr/bin/env python
# coding: utf-8

# # Character-Based RNN
# Work based on [Machine Learning Mastery Tutorial](https://machinelearningmastery.com/develop-character-based-neural-language-model-keras/)

# In[ ]:


from numpy import array
from pickle import dump
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pickle import load
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import random
import string
import os
from keras.callbacks import EarlyStopping


# In[ ]:


os.listdir('../input/reddit-usernames')


# # Data Preprocessing

# In[ ]:



names =  pd.read_csv('../input/reddit-usernames/users.csv', error_bad_lines = False, encoding='latin-1')


# In[ ]:


names.head()


# In[ ]:


names = names.drop(['n'], axis=1)


# Concatenating 10,000 usernames for easy preprocessing

# In[ ]:


data = names.iloc[0][0]
for i in range(1,10000):
    x = names.iloc[i][0]
    data = data + ' ' + x


# In[ ]:


del names


# In[ ]:


# organize into sequences of characters
length = 5
sequence = list()
for i in range(length, len(data)):
    # select sequence of tokens
    seq = data[i-length:i+1]
    # store
    sequence.append(seq)
print('Total Sequences: %d' % len(sequence))


# Sliding window of length 5

# In[ ]:


print(sequence[7])
print(sequence[8])
print(sequence[9])
print(sequence[10])
print(sequence[11])
print(sequence[12])


# In[ ]:


# unique list of letters in vocabulary
chars = sorted(list(set(data)))
# mapping each character to a unique integer
mapping = dict((c, i) for i, c in enumerate(chars))


# In[ ]:


# vocabulary size
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)


# In[ ]:


sequences = list()
for line in sequence:
    # integer encode line
    encoded_seq = [mapping[char] for char in line]
    # store
    sequences.append(encoded_seq)


# In[ ]:


print(sequences[0])


# In[ ]:


del data
del sequence


# * Splitting data into X and y
# * X is 5 characters and y is the next character

# In[ ]:


sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]


# One-hot encoding sequences

# In[ ]:


sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = array(sequences)
y = to_categorical(y, num_classes=vocab_size)


# In[ ]:


#splitting data into train and test sets. 3/4 train, 1/4 test.
x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.25, shuffle=False, random_state=69)
del X
del y
del sequences


# In[ ]:


# define model
model = Sequential()
model.add(LSTM(64, input_shape=(x_train.shape[1], x_train.shape[2]),return_sequences=True))
model.add(LSTM(64))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())


# In[ ]:


# Configure the checkpoint :
checkpoint = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto', restore_best_weights=True)
callbacks_list = [checkpoint]


# In[ ]:


# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam')
# fit model
history = model.fit(x_train, y_train, epochs=100, batch_size=2500, verbose=1,validation_data=(x_test, y_test),callbacks=callbacks_list)


# In[ ]:


# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Test', 'Validation'], loc='upper right')
plt.show()


# # Inference

# In[ ]:


# generate a sequence of characters with a language model
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
    result = ''
    in_text = seed_text
    # generate a fixed number of characters
    for _ in range(n_chars):
        # encode the characters as integers
        encoded = [mapping[char] for char in in_text]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # one hot encode
        encoded = to_categorical(encoded, num_classes=len(mapping))
        # predict character
        yhat = model.predict_classes(encoded, verbose=0)
        # reverse map integer to character
        out_char = ''
        for char, index in mapping.items():
            if index == yhat:
                out_char = char
                break
        # append to input
        in_text += char
        result += char
    return result


# In[ ]:


#Generate random string as primer for char RNN

def randomString(stringLength):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(stringLength))


# # Results

# In[ ]:


#result
for i in range(2):
    result = generate_seq(model, mapping, 5, randomString(4) + ' ', 10).split()
    for username in result:
        print(username)


# In[ ]:


#result
for i in range(2):
    result = generate_seq(model, mapping, 5, randomString(4) + ' ', 10).split()
    for username in result:
        print(username)


# In[ ]:


#result
for i in range(2):
    result = generate_seq(model, mapping, 5, randomString(4) + ' ', 10).split()
    for username in result:
        print(username)

