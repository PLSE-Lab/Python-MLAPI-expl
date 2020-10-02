#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from keras import metrics

import os
print(os.listdir("../input"))


# **Data converted to dataframe**

# In[2]:


data = pd.read_csv('../input/spam.csv',encoding='latin-1')


# In[3]:


data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data = data.rename(columns={"v1":'label', "v2":'text'})
tags = data["label"]
texts = data["text"]

for k in range(len(tags)):
    if (tags[k]=='ham'):
        tags[k]='normal'
    else:
        tags[k]='spam'

h=data['label']=='spam'
print("Spam sms count" ,len(data[h]))
h=data['label']=='normal'
print("Normal sms count" ,len(data[h]))


# **Redundant spaces cleared**

# In[4]:


print (data.head())
print('***************************************************')
print('Spam Sms Example  :  ',data.text.iloc[2])
print('***************************************************')
print('Normal Sms Example  :  ',data.text.iloc[3])


# In[9]:


## For enumeration up to a maximum of 1000
num_max = 1000

## Tags make 0 and 1
le = LabelEncoder()
tags = le.fit_transform(tags)

## The process of enumerating words
tok = Tokenizer(num_words=num_max)
tok.fit_on_texts(texts)

# Number of word counts
#print(tok.word_docs)


# In[20]:


# Indexing the word
#print(tok.word_index)


# In[21]:


# For example, how to enumerate words
print(texts[1])
print(tok.word_index['ok'],tok.word_index['lar'],tok.word_index['joking'],tok.word_index['wif'],tok.word_index['u'],tok.word_index['oni'])


# In[22]:


## A maximum of 100 words and sentences are maintained
max_len = 100
cnn_texts_seq = tok.texts_to_sequences(texts)
for i in range(len(cnn_texts_seq)):
    if(len(cnn_texts_seq[i])>100):
        print('Word Counts:', len(cnn_texts_seq[i]),'Indeks:',i)


# In[23]:


## A maximum of 100 words and sentences are maintained
## The number of words is made from 100. Missing words are written to 0.

cnn_texts_mat = sequence.pad_sequences(cnn_texts_seq,maxlen=max_len)

## There are 30 words in the second sentence.
## All words are indexed
## The most used 1000 words are taken.
## Less used words are removed.
## If the number of words is less than 100, 0 is added. 
## If the number of words is greater than 100 is deleted
print('***************************************************')
print(texts[2])
print(cnn_texts_mat[2])
print('***************************************************')


# In[24]:


## Number of words 101
## The word sad has been deleted.
## There are 100 words left.
print('***************************************************')
print(texts[2157])
print('***************************************************')
print(cnn_texts_mat[2157])
print('***************************************************')

print('sad index:',tok.word_index['sad'], 'story index:',tok.word_index['story'])


# In[25]:


model = Sequential()
model.add(Embedding(1000,20,input_length=max_len))
model.add(Dropout(0.2))
model.add(Conv1D(64,5,padding='valid',activation='relu',strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['acc'])


# In[26]:


history=model.fit(cnn_texts_mat,tags,batch_size=32,epochs=10,verbose=1,validation_split=0.2)


# In[27]:


import matplotlib.pyplot as plt
epochs = range(1, 11)
acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'b+', label='Acc')
plt.plot(epochs, val_acc, 'bo', label='Val Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


# In[ ]:




