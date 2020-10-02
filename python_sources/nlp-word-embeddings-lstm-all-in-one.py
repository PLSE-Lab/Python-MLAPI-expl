#!/usr/bin/env python
# coding: utf-8

# <font color='blue'><b><i>In this notbook, I am using 'sms spam collection' dataset which classifies the messages as spam/ham. I have created a base model using the following techniques.</i></b></font>
# 
# 
# * NLP text preprocessing using NLTK
# * Created static word embeddings using word2vec from Gensim
# * Created LSTM network
# 

# <img src="https://inttix.ai/wp-content/uploads/2019/10/capture-1.jpg" width=600 height=20 />
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
import numpy as np
import string
import os
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import gensim
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

string.punctuation
stopword = nltk.corpus.stopwords.words('english')

lemmatizer = WordNetLemmatizer()
ps = nltk.PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')


# In[ ]:


maxlen = 80
batch_size = 32


# In[ ]:


data = pd.read_csv("/kaggle/input/smsspamcollectiontsv/SMSSpamCollection.tsv", sep='\t')
data.columns = ['label', 'body_text']
data.head()


# In[ ]:


data['label'] = (data['label']=='spam').astype(int)
data.head()


# In[ ]:


data.shape


# ***Step 1 - Cleaning the text data.***
# 
# Removing punctuations, stopwords. Tokenizing the sentences and lemmatizing the words to their original form.

# In[ ]:


def clean_text(text):
    
    ''' Text preprocessing '''

    tokens = tokenizer.tokenize(text.lower())
    
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    tokens = [word for word in stripped if word.isalpha()]

    text = [lemmatizer.lemmatize(word) for word in tokens if word not in stopword]
    return text


# In[ ]:


data['body_text'] = data['body_text'].apply(lambda x: clean_text(x))
body_text_data = data['body_text'].values.tolist()


# In[ ]:


print(len(body_text_data))
body_text_data[0]


# ***Step 2 - Creating word embeddings on cleaned text data using word2vec and saving the model***
# 

# In[ ]:


# word embedding using word2vec
model = gensim.models.Word2Vec(body_text_data, size=100, window=5, min_count=3)
len(model.wv.vocab)


# In[ ]:


# similarity
model.most_similar('customer')


# In[ ]:


# save model
model.wv.save_word2vec_format("spam_word2vec_model.txt", binary=False)


# ***Loading the word embeddings***

# In[ ]:


# Load embeddings

embeddings_index = {}
file = open(os.path.join('', 'spam_word2vec_model.txt'), encoding = "utf-8")

for record in file:
    values = record.split()
    word = values[0]
    coefficient = np.asarray(values[1:])
    embeddings_index[word] = coefficient
file.close()


# In[ ]:


len(embeddings_index)


# In[ ]:


embeddings_index['free']


# ***Padding the data with 0s to have similar length.***

# In[ ]:


tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(body_text_data)
sequences = tokenizer_obj.texts_to_sequences(body_text_data)

word_index = tokenizer_obj.word_index
print("Word index", len(word_index))

X = pad_sequences(sequences, maxlen=maxlen)
print("X shape:", X.shape)

y = data['label'].values
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=15, stratify=y)


# In[ ]:


word_index['free']


# In[ ]:


X[0], y[0]


# ***Creating embedding matrix***

# In[ ]:


# Create embedding matrix for words

EMBEDDING_DIM = 100

max_features = len(word_index) + 1
embedding_matrix = np.zeros((max_features, EMBEDDING_DIM))

for word, i in word_index.items():
    if i > max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# In[ ]:


# embeddings for word - 'free'
embedding_matrix[9]


# ***3 - Creating Neural Network***
# 
# * Using word embeddings from word2vec in first layer
# * Building LSTM network with 64 units
# * Adding Dense layers

# In[ ]:


# Base model

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.layers.embeddings import Embedding
from keras.initializers import Constant

print("Build model...")

model = Sequential()

embedding_layer = Embedding(max_features, EMBEDDING_DIM, 
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=maxlen,
                            trainable=False)

model.add(embedding_layer)
model.add(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


# In[ ]:


print("Train...")
model.fit(X_train, y_train, batch_size=batch_size, epochs=30, validation_data=(X_test, y_test), verbose=2)


# In[ ]:


score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)


# In[ ]:


import matplotlib.pyplot as plt

plt.plot(model.history.history['loss'][5:])
plt.plot(model.history.history['val_loss'][5:])
plt.title('Loss over epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.show()


# <font color='blue'><b><i>Here, I have built the base model which can be further improved. Please upvote if you found it helpful... :)</i></b></font>

# In[ ]:




