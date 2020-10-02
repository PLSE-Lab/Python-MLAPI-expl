#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


tokenizer=tf.keras.preprocessing.text.Tokenizer(oov_token="<OOV>")


# In[ ]:


files=os.listdir('../input/bed-time-stories/')
files


# In[ ]:


data=[]
for i in files:
    file=open('../input/bed-time-stories/'+i,'r')
    text=file.read().lower().split('\n')
    for lines in text:
        data.append(lines)


# removing unwanted data and fit the other texts on the tokenizer

# In[ ]:


while('' in data):
    data.remove('')
while(' ' in data):
    data.remove(' ')
tokenizer.fit_on_texts(data)
tokenizer


# In[ ]:


total_words=len(tokenizer.word_index)+1
total_words


# word index for each words in the tokenizer

# In[ ]:


word_index=tokenizer.word_index
print(word_index)


# In[ ]:


input_sequences=[]
for line in data:
    token_list=tokenizer.texts_to_sequences([line])[0]
    for i in range(1,len(token_list)):
        n_gram_sequences=token_list[:i+1]
        input_sequences.append(n_gram_sequences)


# In[ ]:


max_len=max([len(x) for x in input_sequences])
print(max_len)


# In[ ]:


input_sequences


# In[ ]:


padded=tf.keras.preprocessing.sequence.pad_sequences(input_sequences,maxlen=max_len,padding='pre')
padded


# In[ ]:


input_sequences=np.array(padded)
input_sequences


# In[ ]:


train=input_sequences[:,:-1]
labels=input_sequences[:,-1]
labels


# # Data Encoding

# In[ ]:


label_encoding=tf.keras.utils.to_categorical(labels,num_classes=total_words)
label_encoding


# # Model Architecture
# The model consists of two LSTM layers and two dense layers.
# ![LSTM](https://qph.fs.quoracdn.net/main-qimg-a619fdcafcd6dbb2390d2d4c050e205a.webp)

# In[ ]:


model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(total_words, 100, input_length=max_len-1))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150, return_sequences = True)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(100))
model.add(tf.keras.layers.Dense(total_words/2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.Dense(total_words, activation='softmax'))


# In[ ]:


print(label_encoding.shape)
print(train.shape)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


history=model.fit(train,label_encoding,epochs=100)


# In[ ]:


seed_text = input('Enter a line related to story : ').lower()
next_words = int(input('Enter the number of words you need in the story : '))

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_len-1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += ' '+output_word
print(seed_text)


# In[ ]:




