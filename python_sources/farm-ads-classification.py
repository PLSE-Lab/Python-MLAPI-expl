import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, GRU, LSTM, Dropout, Bidirectional
from keras.layers import Conv1D, MaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import train_test_split

import os
print(os.listdir("../input"))


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
    

x_raw = []
y_raw = []
sm = 0

with open('../input/farm-ads', 'r') as f:
    for line in f:
        y_raw.append(line.split()[0])
        x_raw.append(' '.join(i for i in line.split()[1:]))
        sm += len(line.split()[1:])
        
print(len(y_raw))
print(x_raw[0])

y = [1 if i=='1' else 0 for i in y_raw]

# find ratio of 0 to 1 in our data
sm = 0
for i in y_raw:
    if i=='1':
        sm+= 1
print(sm/len(y_raw))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_raw)
x_tokens = tokenizer.texts_to_sequences(x_raw)
vocab_size = len(tokenizer.word_counts)+1
print(vocab_size)

sum([len(x) for x in x_tokens]) //len(x_tokens)

pad = 'post' 
max_len = 300
embedding_size = 75
batch_size = 128
n_epochs = 50
X_pad = pad_sequences(x_tokens, maxlen=max_len, padding=pad, truncating=pad)
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size = 0.1, random_state= 0)

# =============================================================================
##### LSTM MODEL ####
# =============================================================================

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=True, recurrent_dropout=0.2))
model.add(LSTM(60, return_sequences=False, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid', name='Classification'))
model.summary()

from keras import optimizers
# adam = optimizers.Adam(lr=0.004, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#save_best = ModelCheckpoint('FarmAdsDetection.hdf', save_best_only=True, monitor='val_acc', mode='min')
result = model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_split=0.1, verbose=0)

# model.load_weights(filepath='FarmAdsDetection.hdf')
eval_ = model.evaluate(X_test, y_test)
print(eval_[0], eval_[1]) # loss / accuracy

plot_model(result)

# =============================================================================
##### LSTM MODEL with Global MaxPooling ####
# =============================================================================
n_epochs = 20
embedding_size = 60

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid', name='Classification'))
model.summary()

from keras import optimizers
# adam = optimizers.Adam(lr=0.004, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
save_best = ModelCheckpoint('FarmAdsDetection.hdf', save_best_only=True, monitor='val_acc', mode='min')
result = model.fit(X_train, y_train, epochs=n_epochs, batch_size=128, validation_split=0.2, callbacks=[save_best])

# model.load_weights(filepath='FarmAdsDetection.hdf')
eval_ = model.evaluate(X_test, y_test)
print(eval_[0], eval_[1]) # loss / accuracy

plot_model(result)

# =============================================================================
#### Bidirectional LSTM MODEL ####
# =============================================================================
max_len = 600
embedding_size = 300
batch_size = 128
n_epochs = 30
X_pad = pad_sequences(x_tokens, maxlen=max_len, padding=pad, truncating=pad)
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size = 0.1, random_state= 0)

model1 = Sequential()
model1.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
model1.add(Dropout(0.5))
model1.add(Bidirectional(LSTM(80, return_sequences=False)))
model1.add(Dropout(0.5))
model1.add(Dense(1, activation='sigmoid', name='Classification'))
model1.summary()

sgd = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model1.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

save_best = ModelCheckpoint('FarmAdsDetection.hdf', save_best_only=True, monitor='val_acc', mode='max')
results1 = model1.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_split=0.1, callbacks=[save_best])

model.load_weights(filepath='SpamDetection.hdf')
eval_ = model1.evaluate(X_test, y_test)
print(eval_[0], eval_[1]) # loss / accuracy
plot_model(results1)

# =============================================================================
#### GRU MODEL ####
# =============================================================================
pad = 'post' 
max_len = 500
embedding_size = 75
batch_size = 128
n_epochs = 100
X_pad = pad_sequences(x_tokens, maxlen=max_len, padding=pad, truncating=pad)
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size = 0.1, random_state= 0)

model2 = Sequential()
model2.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
model2.add(Dropout(0.4))
model2.add(GRU(80, return_sequences=False))
model2.add(Dropout(0.4))
model2.add(Dense(1, activation='sigmoid', name='Classification'))
model2.summary()

sgd = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
save_best = ModelCheckpoint('FarmAdsDetection.hdf', save_best_only=True, monitor='val_acc', mode='max')
result2 = model2.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_split=0.1, callbacks=[save_best])

model.load_weights(filepath='SpamDetection.hdf')
eval_ = model2.evaluate(X_test, y_test)
print(eval_[0], eval_[1]) # loss / accuracy
plot_model(result2)

# =============================================================================
#### CNN MODEL ####
# =============================================================================

pad = 'post' 
max_len = 500
embedding_size = 75
batch_size = 128
n_epochs = 50
X_pad = pad_sequences(x_tokens, maxlen=max_len, padding=pad, truncating=pad)
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size = 0.1, random_state= 0)

model3 = Sequential()
model3.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
model3.add(Conv1D(128, 3, activation='relu'))
model3.add(MaxPool1D(3))
model3.add(Dropout(0.2))
model3.add(Conv1D(128, 3, activation='relu'))
model3.add(MaxPool1D(3))
model3.add(GlobalMaxPooling1D())
model3.add(Dropout(0.2))
model3.add(Dense(64, activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(32, activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(1, activation='sigmoid'))
model3.summary()

model3.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
save_best = ModelCheckpoint('FarmAdsDetection.hdf', save_best_only=True, monitor='val_acc', mode='max')
result3 = model3.fit(X_train, y_train, batch_size = batch_size, epochs=n_epochs, validation_split=0.2, verbose=1, callbacks=[save_best])

model3.load_weights(filepath='FarmAdsDetection.hdf')
eval_ = model3.evaluate(X_test, y_test)
print(eval_[0], eval_[1]) # loss / accuracy

plot_model(result3)

# =============================================================================
#### CNN MODEL with GLOVE ####
# =============================================================================

pad = 'post' 
max_len = 500
embedding_size = 100
batch_size = 128
n_epochs = 180
X_pad = pad_sequences(x_tokens, maxlen=max_len, padding=pad, truncating=pad)
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size = 0.1, random_state= 0)

word2vec = {}
f = open('glove.6B/glove.6B.100d.txt', encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    vec = np.asarray(values[1:], dtype='float32')
    word2vec[word] = vec
    
print(len(word2vec))

embedding_matrix = np.zeros((vocab_size, embedding_size))

for word,i in  tokenizer.word_index.items():
    if i < vocab_size:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
         # words not in the glove will be set to zero   
            embedding_matrix[i] = embedding_vector
            
n_epochs = 50
model4 = Sequential()
model4.add(Embedding(input_dim=vocab_size, output_dim=embedding_size,weights = [embedding_matrix], input_length=max_len, trainable=True))
model4.add(Conv1D(128, 3, activation='relu'))
model4.add(MaxPool1D(3))
model4.add(Dropout(0.2))
model4.add(Conv1D(128, 3, activation='relu'))
model4.add(GlobalMaxPooling1D())
model4.add(Dropout(0.2))
model4.add(Dense(64, activation='relu'))
model4.add(Dropout(0.2))
model4.add(Dense(32, activation='relu'))
model4.add(Dropout(0.2))
# =============================================================================
# model4.add(Dense(16, activation='relu'))
# model4.add(Dropout(0.2))
# =============================================================================
model4.add(Dense(1, activation='sigmoid'))
model4.summary()

model4.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
save_best = ModelCheckpoint('FarmAdsDetection.hdf', save_best_only=True, monitor='val_acc', mode='max')
result4 = model4.fit(X_train, y_train, batch_size = batch_size, epochs=n_epochs, validation_split=0.2, verbose=1,  callbacks=[save_best])
eval_ = model4.evaluate(X_test, y_test)
print(eval_[0], eval_[1]) # loss / accuracy

plot_model(result4)

# =============================================================================
# corpus : 43625 words

# Results
# 
# Model                Accuracy
 
# LSTM  :              92% 
# LSTM with MaxPooling 92.2%
# Bidirectional LSTM : 93 % 
# GRU :                90.4%              
# CNN :                92.4%
# Glove-CNN :          91.8%
# =============================================================================



