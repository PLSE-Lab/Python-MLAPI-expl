#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


max_features = 50000


# In[ ]:


train_data = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')
print(train_data.shape)
train_data.head()


# In[ ]:


#toxic
sns.countplot(train_data.toxic)


# In[ ]:


#severe_toxic
sns.countplot(train_data.severe_toxic)


# In[ ]:


#obscene
sns.countplot(train_data.obscene)


# In[ ]:


#threat
sns.countplot(train_data.threat)


# In[ ]:


#insult
sns.countplot(train_data.insult)


# In[ ]:


#identity_hate
sns.countplot(train_data.identity_hate)


# In[ ]:


train_text = train_data['comment_text'].values


# In[ ]:


import nltk
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english')) 

t_data = list()

for i in range(len(train_text)):
    
    if i % 10000 == 0:
        print(i)

    words = nltk.word_tokenize(train_text[i])
    
    words = [word for word in words if word not in stop_words] 

    words=[word.lower() for word in words if word.isalpha()]
    
    # remove single character

    words = [word for word in words if len(word) > 1]
    
    t_data.append(words)


# In[ ]:


from nltk.stem import WordNetLemmatizer 
  
lemmatizer = WordNetLemmatizer()

data_l = list()
for i in range(len(t_data)):
    temp = list()
    for j in t_data[i]:
        temp.append(lemmatizer.lemmatize(j))
    data_l.append(temp)


# In[ ]:


vocab = list()

for i in data_l:
    for j in i:
        vocab.append(j)
# no of words in text
len(vocab)


# In[ ]:


# no of unique words

vocab = set(vocab)
len(vocab)


# In[ ]:


from keras.preprocessing.text import Tokenizer
# function to build a tokenizer

def tokenization(lines):
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(lines)
    return tokenizer

eng_tokens = tokenization(data_l)
eng_vocab_size = len(eng_tokens.word_index) + 1
print('English Vocabulary Size: %d' % eng_vocab_size)


# In[ ]:


m = list()
for i in range(len(data_l)):
    m.append(len(data_l[i]))
plt.plot(m)


# In[ ]:


from keras.preprocessing.sequence import pad_sequences
# encode and pad sequences
def encode_sequences(tokenizer,length,lines):
    # integer encode sequences
    seq = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    seq = pad_sequences(seq, maxlen=length, padding='post')
    return seq

seq_data = encode_sequences(eng_tokens,400,data_l)


# In[ ]:


EMBEDDING_FILE = '../input/glove6b50dtxt/glove.6B.50d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

print('Found %s word vectors.' % len(embeddings_index))


# In[ ]:


all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = eng_tokens.word_index
nb_words = min(eng_vocab_size - 1, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= eng_vocab_size - 1: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# In[ ]:


targets = train_data.iloc[:,2:].values


# In[ ]:


targets.shape


# In[ ]:


seq_data.shape


# In[ ]:


from keras.models import Sequential
from keras.layers import *

model = Sequential()
model.add(Embedding(eng_vocab_size - 1,
                    embed_size,
                    weights=[embedding_matrix],
                    input_length=400,
                    trainable=True))
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(CuDNNGRU(128, return_sequences=True)))
model.add(Bidirectional(CuDNNGRU(64)))
model.add(Dropout(0.25))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(6, activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


# In[ ]:


batch_size = 64
history = model.fit(seq_data, targets, epochs=3, batch_size=batch_size, verbose=1, validation_split=0.1)


# In[ ]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[ ]:


test_data = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')
print(test_data.shape)
test_data.head()


# In[ ]:


test_text = test_data['comment_text'].values


# In[ ]:


t_data_test = list()

for i in range(len(test_text)):
    
    if i % 10000 == 0:
        print(i)

    words = nltk.word_tokenize(test_text[i])
    
    words = [word for word in words if word not in stop_words] 

    words=[word.lower() for word in words if word.isalpha()]
    
    # remove single character

    words = [word for word in words if len(word) > 1]
    
    t_data_test.append(words)


# In[ ]:


data_l_test = list()
for i in range(len(t_data_test)):
    temp = list()
    for j in t_data_test[i]:
        temp.append(lemmatizer.lemmatize(j))
    data_l_test.append(temp)


# In[ ]:


seq_data_test = encode_sequences(eng_tokens,400,data_l_test)


# In[ ]:


pred = model.predict(seq_data_test,verbose=1)


# In[ ]:


pred[0]


# In[ ]:


sam_sub = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')
sam_sub.head()


# In[ ]:


pred[0]


# In[ ]:


_id = test_data['id'].values


# In[ ]:


_id = _id.reshape(-1,1)


# In[ ]:


output = np.array(np.concatenate((_id, pred), 1))


# In[ ]:


output.shape


# In[ ]:


train_data.columns


# In[ ]:


output = pd.DataFrame(output,columns = ['id', 'toxic', 'severe_toxic', 'obscene', 'threat',
       'insult', 'identity_hate'])


# In[ ]:


output.to_csv('submission.csv',index = False)

