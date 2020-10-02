#!/usr/bin/env python
# coding: utf-8

# A simple neural network model using BiLSTM and pre-trained Glove embeddings.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("Train shape : ",train.shape)
print("Test shape : ",test.shape)


# In[ ]:


import nltk


# In[ ]:


from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


# In[ ]:


import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

question_lines = list()
lines = train['question_text'].values.tolist()

for line in lines:
    tokens = word_tokenize(line)
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('','',string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    question_lines.append(words)


# In[ ]:


len(question_lines)


# In[ ]:


max_length = 60
EMBEDDING_DIM = 200
max_features = 50000


# In[ ]:


from tqdm import tqdm

embeddings_index={}
f = open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt')
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:])
    embeddings_index[word] = coefs
f.close()


# In[ ]:


tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(question_lines)
sequences = tokenizer_obj.texts_to_sequences(question_lines)

word_index = tokenizer_obj.word_index
print(len(word_index))
question_pad = pad_sequences(sequences,maxlen=max_length)
target = train['target'].values
print(question_pad.shape)
print(target.shape)


# In[ ]:


num_words = len(word_index)+1
embedding_matrix = np.zeros((num_words,EMBEDDING_DIM))
for word, i in word_index.items():
    if i>num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# In[ ]:


from keras.models import Sequential
from keras.layers import CuDNNLSTM, Dense, Bidirectional


# In[ ]:


model = Sequential()
model.add(Embedding(num_words, EMBEDDING_DIM, input_length=max_length, weights=[embedding_matrix], trainable=False))
model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True))
model.add(Bidirectional(CuDNNLSTM(64)))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model1.summary())


# In[ ]:


VALIDATION_SPLIT = 0.1

indices = np.arange(question_pad.shape[0])
np.random.shuffle(indices)
question_pad = question_pad[indices]
target = target[indices]
num_validation_samples = int(VALIDATION_SPLIT*question_pad.shape[0])

X_train_pad = question_pad[:-num_validation_samples]
y_train = target[:-num_validation_samples]
X_test_pad = question_pad[-num_validation_samples:]
y_test = target[-num_validation_samples:]


# In[ ]:


from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)


# In[ ]:


model.fit(X_train_pad,y_train,batch_size=128,epochs=25,validation_data=(X_test_pad,y_test),verbose=1,callbacks=[es])


# In[ ]:


test.head()


# In[ ]:


question_lines_test = list()
lines_test = test['question_text'].values.tolist()

for line in lines_test:
    tokens = word_tokenize(line)
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('','',string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    question_lines_test.append(words)


# In[ ]:


len(question_lines_test)


# In[ ]:


tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(question_lines_test)
sequences_test = tokenizer_obj.texts_to_sequences(question_lines_test)

word_index_test = tokenizer_obj.word_index
print(len(word_index_test))
question_pad_test = pad_sequences(sequences_test,maxlen=max_length)
print(question_pad_test.shape)


# In[ ]:


target_test = model.predict(question_pad_test)


# In[ ]:


target_test = (target_test>0.5).astype(int)
out_df = pd.DataFrame({"qid":test["qid"].values})
out_df['prediction'] = target_test
out_df.to_csv("submission.csv", index=False)

