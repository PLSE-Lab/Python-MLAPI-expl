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
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout,Embedding,CuDNNLSTM,Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Reading the dataset

# In[ ]:


df = pd.read_csv('../input/train.csv')


# In[ ]:


df.head()


# **Displaying the count of each class in Y label**

# In[ ]:


print(df['target'].value_counts())
sns.countplot(df['target'])


# In[ ]:


x = df['question_text']
y = df['target']


# In[ ]:


token = Tokenizer()


# **Converting the text into sequence for processing in LSTM Layers**

# In[ ]:


token.fit_on_texts(x)
seq = token.texts_to_sequences(x)


# In[ ]:


pad_seq = pad_sequences(seq,maxlen=300)


# In[ ]:


vocab_size = len(token.word_index)+1


# In[ ]:


x = df['question_text']
y = df['target']


# **Using word embeddings so that words with similar words have similar representation in vector space. It represents every word as a vector. The words which have similar meaning are place close to each other.**

# In[ ]:


embedding_vector = {}
f = open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt')
for line in tqdm(f):
    value = line.split(' ')
    word = value[0]
    coef = np.array(value[1:],dtype = 'float32')
    embedding_vector[word] = coef


# **Converting the words in our Vocabulary to their corresponding embeddings and placing them in a matrix.**

# In[ ]:


embedding_matrix = np.zeros((vocab_size,300))
for word,i in tqdm(token.word_index.items()):
    embedding_value = embedding_vector.get(word)
    if embedding_value is not None:
        embedding_matrix[i] = embedding_value


# **Building a LSTM model. LSTM networks are useful in sequence data as they are capable of remembering the past words which help them in understanding the meaning of the sentence which helps in text classification. 
# Bidirectional Layer is helpful as it helps in understanding thesentence from start to end and also from end to start. It works in both the direction. This is useful as the reverse order LSTM layer is capable of learning patterns which are not possible for the normal LSTM layers which goes from start to end of the sentence in the normal order. Hence Bidirectional layers are useful in text classification problems as different patterns can be captured from 2 directions.**

# In[ ]:


model = Sequential()


# In[ ]:


model.add(Embedding(vocab_size,300,weights = [embedding_matrix],input_length=300,trainable = False))


# In[ ]:


model.add(Bidirectional(CuDNNLSTM(75)))


# In[ ]:


model.add(Dense(32,activation = 'relu'))


# In[ ]:


model.add(Dense(1,activation = 'sigmoid'))


# In[ ]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics = ['accuracy'])


# In[ ]:


history = model.fit(pad_seq,y,epochs = 5,batch_size=256,validation_split=0.2)


# In[ ]:


values = history.history


# In[ ]:


val_loss = values['val_loss']
training_loss = values['loss']
training_acc = values['acc']
validation_acc = values['val_acc']
epochs = range(5)

plt.plot(epochs,val_loss,label = 'Validation Loss')
plt.plot(epochs,training_loss,label = 'Training Loss')
plt.title('Epochs vs Loss')
plt.legend()
plt.show()


# In[ ]:


plt.plot(epochs,validation_acc,label = 'Validation Accuracy')
plt.plot(epochs,training_acc,label = 'Training Accuracy')
plt.title('Epochs vs Accuracy')
plt.legend()
plt.show()


# In[ ]:


testing = pd.read_csv('../input/test.csv')


# In[ ]:


testing.head()


# In[ ]:


x_test = testing['question_text']


# In[ ]:


x_test = token.texts_to_sequences(x_test)


# In[ ]:


testing_seq = pad_sequences(x_test,maxlen=300)


# In[ ]:


predict = model.predict_classes(testing_seq)


# In[ ]:


testing['label'] = predict


# In[ ]:


testing.head()


# In[ ]:


submit_df = pd.DataFrame({"qid": testing["qid"], "prediction": testing['label']})
submit_df.to_csv("submission.csv", index=False)


# In[ ]:




