#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
I wil use a simple RNN architecture withData with preprocessed version with no Embedding weights.

"""

import numpy as np
import pandas as pd
from keras.layers import Dense, Input, LSTM, Bidirectional, Conv1D
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, SpatialDropout1D,Dropout,Flatten
from keras.layers import Dropout, Embedding
from keras.preprocessing import text, sequence
from keras.models import Model

train_x = pd.read_csv('../input/cleaned-toxic-comments/train_preprocessed.csv').fillna(" ")
test_x = pd.read_csv('../input/cleaned-toxic-comments/test_preprocessed.csv').fillna(" ")






# In[ ]:





# In[ ]:



max_features=100000
maxlen=150
embed_size=300

train_x['comment_text'].fillna(' ')
test_x['comment_text'].fillna(' ')
train_y = train_x[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
train_x = train_x['comment_text'].str.lower()

test_x = test_x['comment_text'].str.lower()


# In[ ]:


# Vectorize text 
tokenizer = text.Tokenizer(num_words=max_features, lower=True)
tokenizer.fit_on_texts(list(train_x))

train_x = tokenizer.texts_to_sequences(train_x)
test_x = tokenizer.texts_to_sequences(test_x)

train_x = sequence.pad_sequences(train_x, maxlen=maxlen)
test_x = sequence.pad_sequences(test_x, maxlen=maxlen)


# In[ ]:




# Build Model
inp = Input(shape=(maxlen,))

x = Embedding(max_features, embed_size, trainable=True)(inp)
x = Dropout(0.3)(x)

x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.15, recurrent_dropout=0.15))(x)
x = Flatten()(x)
out = Dense(6, activation='softmax')(x)

model = Model(inp, out)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:



# Prediction
batch_size = 32
epochs = 2

model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=1)


# In[ ]:


predictions = model.predict(test_x, batch_size=batch_size, verbose=1)

submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')
submission[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] = predictions
submission.to_csv('submission.csv', index=False)

