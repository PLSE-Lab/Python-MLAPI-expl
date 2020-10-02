#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This is a not very sophisticated approach to text classification employing word embeddings (from GloVe) and a keras ANN.
# 
# Interesting because it performs better than it has any right to, given the little or no attention I've paid to engineering features from the text itself.
# 
# Credit to Jason Brownlee and his fabulous [site](https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/). This is just a hack of his code, but I don't think he'll mind ;-)

# # 1. Read data
# Read the data from CSV and apply some basic pre-processing (remove non-ascii characters, convert our target variable to an integer label).

# In[ ]:


import numpy as np
import pandas as pd
from string import printable
st = set(printable)
data = pd.read_csv("../input/sms-spam-collection-dataset/spam.csv",
                   names=["labelStr","text"],
                   skiprows=1,
                   usecols=[0,1],
                   encoding="latin-1")

data["text"] = data["text"].apply(lambda x: ''.join(["" if  i not in st else i for i in x]))
data["label"]=data["labelStr"].apply(lambda x: 1 if x == "spam" else 0)

docs = data["text"].values
labels = data["label"].values

print(len(docs))


# # 2. Preprocessing
# Tokenize text, convert words / tokens to indexed integers.
# Take each document and convert to a sequence of max length 20 (pad with zeroes if shorter).

# In[ ]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(docs)
print(vocab_size)
# pad documents to a max length of 4 words
max_length = 20
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(len(padded_docs))


# # 3. Import embeddings
# The clever part: import a dictionary of word embeddings that translates each word into a 100 dimensional vector.  
# More info on the project that created this dataset [here](https://nlp.stanford.edu/projects/glove/).  
# The dataset itself was already available on [Kaggle](https://www.kaggle.com/rtatman/glove-global-vectors-for-word-representation) which makes life somewhat easier.

# In[ ]:


# load the whole embedding into memory
embeddings_index = dict()
f = open("../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))


# Very nice, but we only need the subset of these 400,000 words that appear in our docs:

# In[ ]:


# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print(embedding_matrix.shape)


# # 4. Network architecture
# This is wonderfully simple:
#   1. Embedding layer has to come first and takes the embedding matrix and input_length arguments we specified earlier. I've set this to be not trainable to begin with; we'll trust that Stanford have done a better job with their mega dataset than we will with our text messages.
#   2. A flattening layer (translates the outputs of the embedding into a vector of length 2,000: sequence length of 20 x embedding width of 100).
#   3. Sigmoid output node to output probability estimate of class membership.
# 
# We train the network using rmsprop with log-loss as the loss metric.

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding
# define the model

model = Sequential()
model.add(Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=20, trainable=False))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())


# # 5. Training and Evaluation
# Is it any good? Let's find out.  
# Divide our dataset using a holdout strategy:

# In[ ]:


# split dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(padded_docs, labels, test_size=0.2, random_state=42)


# Train the model for ten epochs using the training data.  
# Evaluate using the test data.

# In[ ]:


# fit the model
model.fit(X_train, y_train, epochs=10, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %f' % (accuracy*100))


# OK, it's nowhere near as good as [Jessica's](https://www.kaggle.com/jessicayung) [example](https://www.kaggle.com/jessicayung/word-char-count-top-words-xgboost-1-0-test) using xgboost, but then she put some thought into what features she ought to use!  
