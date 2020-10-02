#!/usr/bin/env python
# coding: utf-8

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


def read_file(filepath):
    with open(filepath) as f:
        str_text = f.read()
        
    return str_text


# In[ ]:


read_file("../input/mobydick/moby_dick_four_chapters.txt")


# In[ ]:


import spacy


# In[ ]:


nlp = spacy.load('en',disable=['parser','tagger','ner'])


# In[ ]:


def seperate_punc(doc_text):
    return [token.text.lower() for token in nlp(doc_text) if token.text not in '\n\n\n\n\n\n\n - \n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n ']


# In[ ]:


p = read_file('../input/w-pustyni/1561_w_pustyni_i_w_puszczy.txt')


# In[ ]:


tokens = seperate_punc(p)


# In[ ]:


len(tokens)


# In[ ]:


#25 words --. network predict #26


# In[ ]:


train_len = 25 + 1

text_seq = []

for i in range(train_len,len(tokens)):
    seq = tokens[i-train_len:i]
    
    text_seq.append(seq)


# In[ ]:


text_seq[100]


# In[ ]:


from keras.preprocessing.text import Tokenizer


# In[ ]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_seq)


# In[ ]:


sequences = tokenizer.texts_to_sequences(text_seq)


# In[ ]:


sequences[1]


# In[ ]:


vocabulary_size = len(tokenizer.word_counts)


# In[ ]:


vocabulary_size


# In[ ]:


sequences = np.array(sequences)


# In[ ]:


from keras.utils import to_categorical


# In[ ]:


X = sequences[:,:-1]


# In[ ]:


y = sequences[:,-1]


# In[ ]:


y = to_categorical(y,num_classes = vocabulary_size+1)


# In[ ]:


seq_len = X.shape[1]


# In[ ]:


from keras.models import Sequential


# In[ ]:


from keras.layers import Dense,LSTM,Embedding


# In[ ]:


def create_model(vocabulary_size, seq_len):
    
    model = Sequential()
    model.add(Embedding(vocabulary_size,seq_len,input_length=seq_len))
    model.add(LSTM(seq_len*3,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(50,activation='relu'))
    
    model.add(Dense(vocabulary_size, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics = ['accuracy'])
    
    model.summary()
    
    return model


# In[ ]:


model = create_model(vocabulary_size+1,seq_len)


# In[ ]:


from pickle import dump,load


# In[ ]:


model.fit(X,y,batch_size=128,epochs=240,verbose=1)


# In[ ]:





# In[ ]:


from keras.preprocessing.sequence import pad_sequences


# In[ ]:


def generate_text(model,tokenizer,seq_len,seed_text,num_gen_words):
    
    output_text = []
    
    input_text = seed_text
    
    for i in range(num_gen_words):
        
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]
        
        pad_encoded = pad_sequences([encoded_text],maxlen=seq_len, truncating='pre')
        
        pred_word_ind = model.predict_classes(pad_encoded, verbose=0)[0]
        
        pred_word = tokenizer.index_word[pred_word_ind]
        
        input_text += ' '+pred_word
        
        output_text.append(pred_word)
    
    return ' '.join(output_text)


# In[ ]:


text_seq[0]


# In[ ]:


import random

random.seed(101)

random_pick = random.randint(0,len(text_seq))

random_text= text_seq[random_pick]


# In[ ]:


random_text


# In[ ]:


seed_text = ' '.join(random_text)


# In[ ]:


seed_text


# In[ ]:


generate_text(model,tokenizer,seq_len,seed_text,num_gen_words=25)


# In[ ]:




