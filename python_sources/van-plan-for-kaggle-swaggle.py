#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt
print(os.listdir("../input"))


# In[ ]:


from keras import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Flatten, Dense, Bidirectional, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split


# In[ ]:


train = pd.read_json('../input/train.json')
train, train_val = train_test_split(train)
test = pd.read_json('../input/test.json')
sample_submission = pd.read_csv('../input/sample_submission.csv')


# In[ ]:



train_train, train_val = train_test_split(train)
xtrain = [k for k in train_train['audio_embedding']]
ytrain = train_train['is_turkey'].values

xval = [k for k in train_val['audio_embedding']]
yval = train_val['is_turkey'].values

# Pad the audio features so that all are "10 seconds" long
x_train = pad_sequences(xtrain, maxlen=10)
x_val = pad_sequences(xval, maxlen=10)

y_train = np.asarray(ytrain)
y_val = np.asarray(yval)


# In[ ]:


model = Sequential()
model.add(BatchNormalization(input_shape=(10, 128)))
model.add(Bidirectional(LSTM(128, dropout=0.4, recurrent_dropout=0.4, activation='relu', return_sequences=True)))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# In[ ]:


#Define a basic LSTM model
# model = Sequential()
# model.add(BatchNormalization(input_shape=(10, 128)))
# model.add(Dropout(.5))
# model.add(Bidirectional(LSTM(128, activation='relu')))
# model.add(Dense(1, activation='sigmoid'))

#maybe there is something better to use, but let's use binary_crossentropy
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])


# In[ ]:


#fit on a portion of the training data, and validate on the rest
model.fit(x_train, y_train,
          batch_size=300,
          epochs=16,
          validation_data=(x_val, y_val))


# In[ ]:


# Get accuracy of model on validation data. It's not AUC but it's something at least!
score, acc = model.evaluate(x_val, y_val, batch_size=300)
print('Test accuracy:', acc)


# In[ ]:


test_data = test['audio_embedding'].tolist()
submission = model.predict(pad_sequences(test_data))
submission = pd.DataFrame({'vid_id':test['vid_id'].values,'is_turkey':[x for y in submission for x in y]})
submission['is_turkey'] = submission.is_turkey.round(0).astype(int)
print(submission.head(40))
submission.to_csv('submission.csv', index=False)


# In[ ]:




