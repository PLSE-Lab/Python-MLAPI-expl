# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Dropout,Bidirectional
from sklearn.cross_validation import train_test_split
import re
import numpy as np
import os
import json
from keras.layers import Conv1D,MaxPooling1D
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import pandas as pd
import os
#print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.

df= pd.read_csv("../input/NationalNames.csv")
df = df[['Name','Gender']]

df['Name'] = df['Name'].apply(lambda x: x.lower())


print("Data size "+str(df['Name'].count()))
df=df.drop_duplicates(['Name'], keep='last')
print("Data size "+str(df['Name'].count()))

max_features =1000
tokenizer = Tokenizer(num_words=max_features ,char_level=True)
tokenizer.fit_on_texts(df['Name'].values)

dictionary = tokenizer.word_index
# Let's save this out so we can use it later
with open('wordindex.json', 'w') as dictionary_file:
    json.dump(dictionary, dictionary_file)

X = tokenizer.texts_to_sequences(df['Name'].values)
X = pad_sequences(X,20)
dummies = pd.get_dummies(df['Gender'])
print(dummies.head())
Y=dummies.values

# LSTM model
embed_dim =64
lstm_out =128

model = Sequential()
model.add(Embedding(max_features, embed_dim,input_length =  X.shape[1]))
model.add(LSTM(lstm_out, return_sequences=True))
model.add(LSTM(lstm_out, return_sequences=True))
model.add(LSTM(lstm_out, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

# define early stopping callback
earlystop = EarlyStopping(patience=5)
callbacks_list = [earlystop]


batch_size = 2000
history=model.fit(X, Y, validation_split=0.33, callbacks=callbacks_list,epochs=100, batch_size=batch_size)
model.save('gender_lstm_model.h5')
