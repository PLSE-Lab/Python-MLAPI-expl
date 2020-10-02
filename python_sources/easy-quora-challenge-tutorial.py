#!/usr/bin/env python
# coding: utf-8

# ## Easy Beginner Quora Tutorial
# 
# In this notebook will explain the main components of the Quora kernels you will read.  After reading this notebook you will much better understand some of the key components of the more complicated kernels.  Check links at bottom for helpful links.

# In[ ]:


import numpy as np
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding


# ### Define a simple vocab
# 
# Rather than using the training data containing many thousands of words, we will keep it simple by defining a small vocab and class labels
# 
# How we can learn a word embedding while fitting a neural
# network on a text classification problem. We define a small problem where we have 10
# text documents, each with a comment about a piece of work a student submitted. Each text
# document is classified as positive 1 or negative 0. This is a simple sentiment analysis problem.
# First, we will define the documents and their class labels.

# In[ ]:


# define documents
docs = ['Well done!',
'Good work',
'Great effort',
'nice work',
'Excellent!',
'Weak',
'Poor effort!',
'not good',
'poor work',
'Could have done better.']

# define class labels
labels = np.array([1,1,1,1,1,0,0,0,0,0])


# The first step is to define the examples, encode them as integers,
# then pad the sequences to be the same length. In this case, we need to be able to map words to
# integers as well as integers to words. Keras provides a Tokenizer class that can be fit on the
# training data, can convert text to sequences consistently by calling the texts to sequences()
# method on the Tokenizer class, and provides access to the dictionary mapping of words to
# integers in a word index attribute

# In[ ]:


# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1

# integer encode the documents
encoded_docs = t.texts_to_sequences(docs)
print(encoded_docs)


# Running the example first prints the integer encoded documents. You can see that the word `work` appears in several of the sentences and is encoded as `1` in all the encoded docs.
# 
# The sequences have different lengths and Keras prefers inputs to be vectorized and all inputs
# to have the same length. We will pad all input sequences to have the length of 4.
# 

# In[ ]:


# pad documents to a max length of 4 words
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)


# Then the padded versions of each document are printed, making them all uniform length

# ### Using Pre-Trained GloVe Embedding
# 
# The Keras Embedding layer can also use a word embedding learned elsewhere. It is common
# in the field of Natural Language Processing to learn, save, and make freely available word
# embeddings. For example, the researchers behind GloVe method provide a suite of pre-trained
# word embeddings on their website released under a public domain license
# 
# In the Quora challenge glove.840B.300d.txt is commonly used. It has a vector size of 300 dimensions - the largest available.

# For example, below are the first line of the embedding ASCII text file showing the embedding for `the`

# In[ ]:


#look at the test file
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

count = 0
import csv
with open(EMBEDDING_FILE,) as f:
    reader = csv.reader(f)
    for row in reader:
        if count == 2:
            print (row)
            break
        count +=1
        


# You can see the word `the` and its associated weights.  The neural network will use these weights when it sees this word
# 
# Next, we need to load the entire GloVe word embedding file into memory as a dictionary of
# word to embedding array.

# In[ ]:


def get_coefs(word,*arr): 
    return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE,errors='ignore'))


# In[ ]:


print('Loaded %s word vectors.' % len(embeddings_index))


# Next, we need to create a matrix of one embedding for each word in the training
# dataset. We can do that by enumerating all unique words in the Tokenizer.word index and
# locating the embedding weight vector from the loaded GloVe embedding. The result is a matrix
# of weights only for words we will see during training

# In[ ]:


# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 300))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# ### Keras Embedding Layer
# 
# Keras offers an Embedding layer that can be used for neural networks on text data. It requires
# that the input data be integer encoded, so that each word is represented by a unique integer.
# This data preparation step can be performed using the Tokenizer API also provided with
# Keras.
# 
# Here it is used to load a pre-trained word embedding model from above
# 
# The Embedding layer is defined as the first hidden layer of a network. It must specify 3
# arguments:
# - `input dim`: This is the size of the vocabulary in the text data. For example, if your data
# is integer encoded to values between 0-10, then the size of the vocabulary would be 11
# words
# - `output dim`: This is the size of the vector space in which words will be embedded. It
# defines the size of the output vectors from this layer for each word. For example, it could
# be 32 or 100 or even larger. 
# - `input length`: This is the length of input sequences, as you would define for any input
# layer of a Keras model. For example, if all of your input documents are comprised of 1000
# words, this would be 1000

# For example, below we define an Embedding layer with a vocabulary of 200 (e.g. integer
# encoded words from 0 to 199, inclusive), a vector space of 32 dimensions in which words will be
# embedded, and input documents that have 50 words each.
# 
# e = Embedding(200, 32, input_length=50)
# 

# ### Define a simple Model
# 
# The examples in most kernels use a more complicated neural network known as a LSTM, this is essentially a network that has memory.  Here we use a simple neural network.

# In this example the Embedding layer has weights that are pre-learned - we use the weights from GLoVe word embedding here.   The output of the Embedding layer is a 2D vector with
# one embedding for each word in the input sequence of words (input document). Here we  connect a Dense layer directly to the Embedding layer, so we first flatten the 2D output matrix to a 1D vector using the Flatten layer.

# In[ ]:


# define model
model = Sequential()
e = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=4, trainable=False)
model.add(e)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# summarize the model
model.summary()

# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))


# Given our very small set of words we get perfect accuracy, but this is only a small example - in reailty you will have many thousands of words in the training and test data

# If you liked this kernel then please upvote, thanks :)

# - http://neuralnetworksanddeeplearning.com/chap1.html
# - https://machinelearningmastery.com/introduction-machine-learning-scikit-learn/
# - https://www.kdnuggets.com/2017/07/machine-learning-exercises-python-introductory-tutorial-series.html
# - https://skymind.ai/wiki/lstm
# - https://machinelearningmastery.com/crash-course-deep-learning-natural-language-processing/
# - http://blog.kaggle.com/2017/11/27/introduction-to-neural-networks/?utm_medium=email&utm_source=mailchimp&utm_campaign=newsletter+dec+2017

# In[ ]:




