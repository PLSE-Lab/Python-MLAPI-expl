#!/usr/bin/env python
# coding: utf-8

# In this notebook, we're going to make a simple model to classify an article as promotional or not promotional. This notebook serves as a basic beginner's guide to LSTMs and text processing.

# In[ ]:


import numpy as np
RANDOM_SEED = 4
np.random.seed(RANDOM_SEED) # set random seed for reproducability


# # Import Data

# In[ ]:


import pandas as pd
df_neut = pd.read_csv("../input/wikipedia-promotional-articles/good.csv")
df_prom = pd.read_csv("../input/wikipedia-promotional-articles/promotional.csv")


# In both datasets, we only need the text.

# In[ ]:


df_prom = df_prom.drop(df_prom.columns[1:], axis=1)
df_neut = df_neut.drop(df_neut.columns[1:], axis=1)


# Our data is structured where each row corresponds to the text of one article.

# In[ ]:


df_neut.head()


# In[ ]:


df_prom.head()


# It will be easier to use we combine both into one dataframe. When we do this, we need to add labels corresponding to promotional or neutral as well. Here, I'm using promotional = 1 and neutral = 0.

# In[ ]:


df_neut.insert(1, 'label', 0) # neutral labels
df_prom.insert(1, 'label', 1) # promotional labels


# In[ ]:


df_prom.head()


# In[ ]:


df_neut.head()


# In[ ]:


df = pd.concat((df_neut, df_prom), ignore_index=True, axis=0) # merge dataframes


# Currently our data has all neutral articles first and all promotional articles second. The promotional articles are alphabetically sorted as well. Because of this, we're going to randomize the order.

# In[ ]:


df.head()


# In[ ]:


df = df.reindex(np.random.permutation(df.index))
df.head()


# Prior to training, we need to split the data into training/testing sets as well.

# In[ ]:


from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)


# # Text Processing

# When we process text data, we need to vectorize it, that is, convert it to data that can be interpreted by the model. To do this, we're using the Tokenizer class in Keras.
# 
# Tokenizer preprocesses the words, removing symbols and making the text lowercass, then adds all unique words to a dictionary. When we then vectorize text, that text gets converted to a sequence of indices, corresponding to a word's position in the dictionary.

# In[ ]:


from keras.preprocessing.text import Tokenizer

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000

text_data = [str(txt) for txt in df_train['text'].values] # convert text data to strings
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True) # create tokenizer object
tokenizer.fit_on_texts(text_data) # make dictionary

x_train = tokenizer.texts_to_sequences(text_data) # vectorize dataset


# We need to then convert our data to a fixed shape. The following Keras function will pad each document with a default value of 0 if a given sequence is smaller than the target sequence length and truncate if it is larger.

# In[ ]:


from keras.preprocessing import sequence

# Max number of words in each sequence
MAX_SEQUENCE_LENGTH = 400

x_train = sequence.pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH)


# Getting the test labels as well

# In[ ]:


y_train = df_train['label'].values


# # Creating the Model

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam

model = Sequential()


# **An issue with word representation:**
# 
# Recall that when we vectorized our data, we converted each word into indices. This allows words to be processed, but it creates the issue that the values lose their meaning. In image data, the magnitude of a datapoint corresponds the the brightness of a pixel, so a larger value will correspond to a brighter pixel. However, the magnitude of datapoints in a vectorized dataset don't have that meaning -- a point with value 5000 doesn't necessarily have more meaning than a point with the value 0. If we represent the text as categorical data with one-hot arrays, we get the issue of sparcity: with a vocabulary of 50000 words, where only 400 words are used per document, this method would be extremely inefficient.
# 
# This is why we use an embedding layer. An embedding layer is simply a matrix where each row corresponds to a representation of a word in the vocabulary. Each representation is a 1D vector of real numbers that is learned when training. Now, we can process words with vectors of managable, fixed sizes and, because these representations are learned, they often are semantically reasonable as well.

# ![Embedding Layer](https://miro.medium.com/max/1596/1*1hPDk0gPyIBg0D5SzY5t7Q.png)
# 
# Source: https://towardsdatascience.com/what-the-heck-is-word-embedding-b30f67f01c81

# Embedding layers aren't limited to word representations either, they can be useful in many types of data, especially when it is sparse.

# We create an embedding layer by specifying the input size of the mapping (number of words in vocabulary + 1), output size (length of vector representation), and input length (amount of words per sequence).
# 
# We add 1 to the vocab size because 0 is a reserved value that we use for padding.

# In[ ]:


EMBEDDING_DIM = 100
model.add(Embedding(MAX_NB_WORDS+1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))


# **LSTM Layer**
# 
# Initially, Recurrent Neural Networks (RNNs) allowed for neural networks to effectively process time series data by taking in past outputs as input:
# 
# ![rnn](http://www.easy-tensorflow.com/images/NN/01.png)
# 
# Source: http://www.easy-tensorflow.com/tf-tutorials/recurrent-neural-networks/vanilla-rnn-for-classification
# 
# However, this type of network had a very short memory. It wasn't possible for the network to take into account context information from very far in the past, something very important for processing documents. LSTMs offered a solution through a structure that consists of a cell state, hidden state, and multiple gates:
# 
# ![lstm](https://www.researchgate.net/publication/324600237/figure/fig3/AS:616974623178753@1524109621725/Long-Short-term-Memory-Neural-Network.png)
# 
# Source: https://www.researchgate.net/figure/Long-Short-term-Memory-Neural-Network_fig3_324600237
# 
# The hidden state (\\(h_{t-1}\\)) of an LSTM consists of the output from the previous time step.
# The cell state (\\(C_{t}\\)) passes through each time step, and its alteration is decided by a series of gates. The gates are essentially one layer neural networks that use the current input along with the hidden state to decide what values within the cell state to forget and what new information to add. Another gate also decides what information to output.
# 
# This is an extreme oversimplification and I would recommend anyone interested to check out this [great article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).

# Adding the LSTM layer to the model, we need to set the number of units. This number corresponds to the dimensionality of the output space and thus also the dimensionality of the cell state, the hidden state, and the neural network gates.

# In[ ]:


model.add(LSTM(80))


# To complete the classifier, we can then add two dense layers, the last one being a sigmoid output, producing a range between 0 and 1, corresponding to not promotional and promotional respectively.
# The dropout layer randomly sets a proportion of its input units to zero. This is added to prevent overfitting as it reduces the neural network's dependence on certain features.

# In[ ]:


model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


# Since we're doing binary classification, we're using [binary cross entropy loss](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a). The Adam optimizer is used to train to model.

# In[ ]:


model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])


# # Training

# We don't need to train for too many epochs as this model will overfit fairly easily.

# In[ ]:


EPOCHS = 2
BATCH_SIZE = 64

history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.15)


# # Testing

# Converting the test set using the previously trained tokenizer

# In[ ]:


x_test = np.array(tokenizer.texts_to_sequences([str(txt) for txt in df_test['text'].values]))
x_test = sequence.pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)

y_test = df_test['label'].values


# Evaluating the model

# In[ ]:


scores = model.evaluate(x_test, y_test, batch_size=128)
print("The model has a test loss of %.2f and a test accuracy of %.1f%%" % (scores[0], scores[1]*100))


# This model alone obviously isn't ideal: a couple ideas for improvement are to increase the number of epochs, change the intial learning rate (by setting the parameter "learning_rate" in Adam()), or change the number of units in the LSTM.
