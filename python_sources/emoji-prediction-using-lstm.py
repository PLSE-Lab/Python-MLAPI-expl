#!/usr/bin/env python
# coding: utf-8

# **Numpy**
# 1.   NumPy is the fundamental package for scientific computing with Python.
# 2.   It provides a high-performance multidimensional array object, and tools for    working with these arrays.
# 
# **Pandas**
# 
# 
# 1.   Pandas is the most popular python library that is used for data analysis.
# 2.   We can manipulate like Excel sheets
# 

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


# importing the traning data
train_data = pd.read_csv('../input/emoji-prediction-dataset/Train.csv')
train_data.head()


# In[ ]:


# import the testing data
test_data = pd.read_csv("../input/emoji-prediction-dataset/Test.csv")
test_data.head()


# In[ ]:


# import the mappings file
mappings = pd.read_csv("../input/emoji-prediction-dataset/Mapping.csv")
mappings.head()


# In[ ]:


# print the shapes of all files
train_data.shape, test_data.shape, mappings.shape


# In[ ]:


train_length = train_data.shape[0]
test_length = test_data.shape[0]
train_length, test_length


# **NLTK is a library for Natural Language Processing (NLP) to create features from text**
# When using words as features, we need to handle:
# 1.   Context -> eg: Not good
# 2.   Identify root words -> eg: help, helper, helping
# 3.   Words with similar meaning -> eg: good and nice
# 
# **Stopwords are useless words or commonly used words. They add very little information to our model so can be removed**

# In[ ]:


from nltk.corpus import stopwords


# In[ ]:


stop_words = stopwords.words("english")
stop_words[:5]


# We need to follow the following steps to pre process the data before using it:
# 
# 
# 1.   Each tweet should be tokenized into a list of words
# 2.   Remove words starting with **@** because they generally refer to twitter       handles and thus provide little or no information
# 3.   Remove **stopwords**
# 4.   Remove the **#** character to get the actual word used as hashtag

# In[ ]:


# tokenize the sentences
def tokenize(tweets):
    stop_words = stopwords.words("english")
    tokenized_tweets = []
    for tweet in tweets:
        # split all words in the tweet
        words = tweet.split(" ")
        tokenized_string = ""
        for word in words:
            # remove @handles -> useless -> no information
            if word[0] != '@' and word not in stop_words:
                # if a hashtag, remove # -> adds no new information
                if word[0] == "#":
                    word = word[1:]
                tokenized_string += word + " "
        tokenized_tweets.append(tokenized_string)
    return tokenized_tweets


# **Keras** - It is an Open Source Neural Network library written in Python that runs on top of Tensorflow, i.e., it uses tensors to run the operations. 
# 
# **Tokenizer** - vectorize text by turning each text into a sequence of integers
# 
# **filters** - a string where each element is a character that will be filtered from text
# 
# **lower** - boolean for lower case conversion
# 
# **tokenizer.texts_to_sequences(tweets)** - transform each tweet in tweets to a sequence of integers
# 

# In[ ]:


# translate tweets to a sequence of numbers
def encod_tweets(tweets):
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', split=" ", lower=True)
    tokenizer.fit_on_texts(tweets)
    return tokenizer, tokenizer.texts_to_sequences(tweets)


# **Example**  
# Uncomment and run the following code cell to see the example of the output

# In[ ]:


# example_str = tokenize(['This is a good day. @css #mlhlocalhost'])
# encod_str = encod_tweets(example_str)
# print(example_str)
# print(encod_str)


# **pad_sequences** - transforms list of sequences (list of integers) into 2D numpy arrays of shape (num_samples, maxlen)
# 
# maxlen is the length of longest sequence, can be provided as an argument also
# 
# If sequences are shorter than maxlen, they are padded with value at front or end (pre or post padding)
# If sequences are longer than maxlen, they are truncated
# 
# **bit_vec** -> vector of 0 and 1

# In[ ]:


# apply padding to dataset and convert labels to bitmaps
def format_data(encoded_tweets, max_length, labels):
    x = pad_sequences(encoded_tweets, maxlen= max_length, padding='post')
    y = []
    for emoji in labels:
        bit_vec = np.zeros(20)
        bit_vec[emoji] = 1
        y.append(bit_vec)
    y = np.asarray(y)
    return x, y


# In[ ]:


# create weight matrix from pre trained embeddings
def create_weight_matrix(vocab, raw_embeddings):
    vocab_size = len(vocab) + 1
    weight_matrix = np.zeros((vocab_size, 300))
    for word, idx in vocab.items():
        if word in raw_embeddings:
            weight_matrix[idx] = raw_embeddings[word]
    return weight_matrix


# **Embeddings** -> are used mainly for text processing.
# 
# **Example**:
# 
# Hope to see you soon. -> [0, 1, 2, 3, 4] (embedding of words)
# 
# Nice to see you again. -> [5, 1, 2, 3, 6]
# 
# **Vocab size** = number of unique words in vocabulary = max number in embeddings + 1 = 6 + 1 = 7
# 
# **Sequential** -> means we are using linear stack of layers
# 
# **LSTM** -> Long Short term memory
# 
# **Bidirectional** -> wrapper to indicate the type of LSTM used
# 
# **Dense** -> Denselu connected neural network
# 
# **Activation function** -> decided whether a neuron should be activated or not by calculating weighted sum and adding bias to it.
# 
# It provides non-linearlity to output of neuron
# 
# **A neural network without an activation function is just a Linear Regression**. By using activation function, we can make our model solve complex functions.
# 
# **Softmax** -> similar to **Sigmoid function** -> used for multiple classes, gives output between 0 & 1 and divide by sum of outputs
# 
# **Optimizer** -> finds the trainable variables on which cost depends and change their values to **optimize cost**
# 
# **Entroy** -> -sum(p log p) -> avg amount of information drawn from one sample

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.wrappers import Bidirectional
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# In[ ]:


# final model
def final_model(weight_matrix, vocab_size, max_length, x, y, epochs = 5):
    embedding_layer = Embedding(vocab_size, 300, weights=[weight_matrix], input_length=max_length, trainable=True, mask_zero=True)
    model = Sequential()
    model.add(embedding_layer)
    model.add(Bidirectional(LSTM(128, dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(128, dropout=0.2)))
    model.add(Dense(20, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x, y, epochs = epochs, validation_split = 0.25)
    score, acc = model.evaluate(x_test, y_test)
    return model, score, acc


# Tokenizing the train and test tweets and then encoding them

# In[ ]:


import math


# In[ ]:


tokenized_tweets = tokenize(train_data['TEXT'])
tokenized_tweets += tokenize(test_data['TEXT'])
max_length = math.ceil(sum([len(s.split(" ")) for s in tokenized_tweets])/len(tokenized_tweets))
tokenizer, encoded_tweets = encod_tweets(tokenized_tweets)
max_length, len(tokenized_tweets)


# Apply padding to the encoded data using pad_sequences for both train and test tweets

# In[ ]:


x, y = format_data(encoded_tweets[:train_length], max_length, train_data['Label'])
len(x), len(y)


# In[ ]:


x_test, y_test = format_data(encoded_tweets[train_length:], max_length, test_data['Label'])
len(x_test), len(y_test)


# Building vocabulary using **word_index** 

# In[ ]:


vocab = tokenizer.word_index
vocab, len(vocab)


# **keyedvectors** -> word vector storage and look up
# 
# It is used to load hidden weight matrix
# 
# **binary** -> to specify whether the data is binary or not

# In[ ]:


from gensim.models.keyedvectors import KeyedVectors


# In[ ]:


# load the GloVe vectors in a dictionary:

embeddings_index = {}
f = open('/kaggle/input/glove840b300dtxt/glove.840B.300d.txt','r',encoding='utf-8')
for line in f:
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray([float(val) for val in values[1:]])
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# create the weight matrix using our vocab and embeddings_index

# In[ ]:


weight_matrix = create_weight_matrix(vocab, embeddings_index)
len(weight_matrix)


# Run the final model on train data

# In[ ]:


model, score, acc = final_model(weight_matrix, len(vocab)+1, max_length, x, y, epochs = 5)
model, score, acc


# In[ ]:


model.summary()


# Use .predict() funtion to predict the y values for x_test
# 
# y values are numpy arrays of length 20 == number of classes
# 
# The class can be found out by finding the index of the maximum value

# In[ ]:


y_pred = model.predict(x_test)
y_pred


# In[ ]:


for pred in y_pred:
    print(np.argmax(pred))


# Print the classification report which gives:
# 
# **precision** -> what % of predicted a's are actually a
# 
# **recall** -> what % of a are predicted to be a
# 
# **fi-score** -> Harmonic mean of precision and recall
# 
# **support** -> actual values of each class

# In[ ]:


import math
from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


y_pred = np.array([np.argmax(pred) for pred in y_pred])
y_true = np.array(test_data['Label'])
print(classification_report(y_true, y_pred))


# In[ ]:


emoji_pred = [mappings[mappings['number'] == pred]['emoticons'] for pred in y_pred]
emoji_pred


# In[ ]:


for i in range(100, 150):
    test_tweet = test_data['TEXT'][i]
    pred_label = y_pred[i]
    pred_emoji = emoji_pred[i]
    print('tweet: ', test_tweet)
    print('pred emoji: ', pred_label, pred_emoji)
    print('-'*50)

