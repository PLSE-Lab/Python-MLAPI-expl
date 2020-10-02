#!/usr/bin/env python
# coding: utf-8

# In[61]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[62]:


data = pd.read_csv('../input/train.tsv',sep='\t')


# In[63]:


data.head()


# In[64]:


import seaborn as sns


# In[65]:


sns.countplot(x='Sentiment',data=data)


# In[66]:


# 0 - negative
# 1 - somewhat negative
# 2 - neutral
# 3 - somewhat positive
# 4 - positive


# In[67]:


phrase = data['Phrase']
label = data['Sentiment']


# In[68]:


from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, RepeatVector
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences


# In[69]:


import nltk
nltk.download('punkt')

t_data = list()

for i in range(len(phrase)):
    
    if i % 1000 == 0:
        print(i)

    words = nltk.word_tokenize(phrase[i])

    words=[word.lower() for word in words if word.isalpha()]
    
    # remove single character

    words = [word for word in words if len(word) > 1]
    
    t_data.append(words)


# In[70]:


len(t_data)


# In[71]:


t_data[0]


# In[72]:


nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer 
  
lemmatizer = WordNetLemmatizer()

data_l = list()
for i in range(len(t_data)):
    temp = list()
    for j in t_data[i]:
        temp.append(lemmatizer.lemmatize(j))
    data_l.append(temp)


# In[73]:


vocab = list()

for i in data_l:
    for j in i:
        vocab.append(j)


# In[74]:


len(vocab)


# In[75]:


vocab = set(vocab)
len(vocab)


# In[76]:


from keras.preprocessing.text import Tokenizer
# function to build a tokenizer
def tokenization(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

eng_tokens = tokenization(data_l)
eng_vocab_size = len(eng_tokens.word_index) + 1
print('English Vocabulary Size: %d' % eng_vocab_size)


# In[77]:


m = list()
for i in range(len(data_l)):
    m.append(len(data_l[i]))


# In[78]:


plt.plot(m)


# In[79]:


np.max(m)


# In[80]:


from keras.preprocessing.sequence import pad_sequences
# encode and pad sequences
def encode_sequences(tokenizer,length,lines):
    # integer encode sequences
    seq = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    seq = pad_sequences(seq, maxlen=length, padding='post')
    return seq

seq_data = encode_sequences(eng_tokens,47,data_l)


# In[81]:


from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

Y = data.iloc[:,-1].values
Y = to_categorical(Y)


# In[82]:


Y


# In[83]:


seq_data.shape


# In[84]:


Y.shape


# In[85]:


seq_data = np.array(seq_data)
Y = np.array(Y)


# In[86]:


seq_data


# In[87]:


Y.shape


# In[122]:


from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, CuDNNLSTM, Conv1D, GlobalMaxPool1D, SpatialDropout1D
from keras.layers import Bidirectional

embedding_size=300

model=Sequential()
model.add(Embedding(eng_vocab_size, embedding_size, input_length=47, trainable=False))
model.add(SpatialDropout1D(0.3))
model.add(Bidirectional(CuDNNLSTM(100,return_sequences=True)))
model.add(Dropout(0.2))
model.add(Conv1D(128, 1, strides = 1,  padding='causal', activation='relu'))
model.add(Conv1D(256, 3, strides = 1,  padding='causal', activation='relu'))
model.add(Conv1D(512, 5, strides = 1,  padding='causal', activation='relu'))
model.add(GlobalMaxPool1D())
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))

print(model.summary())


# In[123]:


from keras.callbacks import ModelCheckpoint

# checkpoint
filepath="model_weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.compile(loss='categorical_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])


# In[124]:


batch_size = 64
num_epochs = 20
# X_valid, y_valid = X_train[:batch_size*100], Y_train[:batch_size*100,:]
# X_train2, y_train2 = X_train[batch_size:], Y_train[batch_size:,:]
history = model.fit(seq_data, Y, validation_split=0.1, batch_size=batch_size, epochs=num_epochs, verbose=1, callbacks=callbacks_list)


# In[125]:


model.load_weights('model_weights.hdf5')


# In[ ]:


test_data = pd.read_csv('../input/test.tsv',sep='\t')


# In[ ]:


test_data.head()


# In[ ]:


test_phrase = test_data['Phrase']


# In[ ]:


test_phrase.shape


# In[ ]:


t_data = list()

for i in range(len(test_phrase)):
    
    if i % 1000 == 0:
        print(i)

    words = nltk.word_tokenize(test_phrase[i])

    words=[word.lower() for word in words if word.isalpha()]
    
    # remove single character

    words = [word for word in words if len(word) > 1]
    
    t_data.append(words)


# In[ ]:


t_data = np.array(t_data)
t_data.shape


# In[ ]:


data_l = list()
for i in range(len(t_data)):
    temp = list()
    for j in t_data[i]:
        temp.append(lemmatizer.lemmatize(j))
    data_l.append(temp)
data_l = np.array(data_l)
data_l.shape


# In[ ]:


test_seq_data = encode_sequences(eng_tokens,47,data_l)
test_seq_data.shape


# In[ ]:


sample = pd.read_csv('../input/sampleSubmission.csv')
sample.head()


# In[ ]:


test_seq_data


# In[ ]:


op = model.predict_classes(test_seq_data[0].reshape(-1,47))


# In[ ]:


op


# In[ ]:


pred = list()

for i in range(len(test_seq_data)):
    if i % 1000 == 0:
        print(i)
    x = model.predict_classes(test_seq_data[i].reshape(-1,47))
    pred.append(x)


# In[ ]:


pred = np.array(pred)
pred.shape


# In[ ]:


p_id = test_data['PhraseId']
p_id = np.array(p_id).reshape(66292, 1)
p_id.shape


# In[ ]:


output = np.array(np.concatenate((p_id, pred), 1))

output = pd.DataFrame(output,columns = ["PhraseId","Sentiment"])

output.to_csv('out.csv',index = False)

