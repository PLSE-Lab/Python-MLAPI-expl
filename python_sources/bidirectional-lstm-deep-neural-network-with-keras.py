import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Embedding
from keras.layers import Bidirectional, GlobalMaxPool1D, Conv1D
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Set up callbacks
tensorboard = TensorBoard(log_dir='./logs')
early_stopping = EarlyStopping(monitor='val_acc', patience=2)
checkpoint = ModelCheckpoint('save/model.checkpoint.h5',
                                monitor='val_acc',
                                verbose=1,
                                save_best_only=True,
                                mode='max')

# Load the training and test data
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
train_data = train_data.fillna(' ')
test_data = test_data.fillna(' ')
y = train_data['label'].values # Class values

train_sent = train_data['title'] + ' ' + train_data['author'] + ' ' + train_data['text']
train_sent = train_sent.dropna(how='all')
test_sent = test_data['title'] + ' ' + test_data['author'] + ' ' + test_data['text']

tokenizer = Tokenizer(num_words=20000, lower=True)
tokenizer.fit_on_texts(list(train_sent))
train_tokens = tokenizer.texts_to_sequences(train_sent)
test_tokens = tokenizer.texts_to_sequences(test_sent)

train = pad_sequences(train_tokens, maxlen=1000)
test = pad_sequences(test_tokens, maxlen=1000)

model = Sequential()
model.add(Embedding(20000, 256, input_length=1000))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform"))
model.add(GlobalMaxPool1D())
model.add(Dropout(0.1))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train, y, batch_size=32, epochs=20, validation_split=0.1, callbacks=[checkpoint, early_stopping, tensorboard])

# Make predictions
results = model.predict(test)
results = np.round(results)
results = results.reshape(results.shape[0])
results = pd.Series(results,name="label")
prediction = pd.concat([pd.Series(range(20800, 26000), name='id'),results], axis=1)
prediction['label'] = prediction['label'].map(int)
prediction.to_csv("submission.csv", index=False)