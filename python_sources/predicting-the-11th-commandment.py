#!/usr/bin/env python
# coding: utf-8

# # Predicting the 11th Commandment
# 
# ## Introduction
# 
# This notebook is for me to get to know Recurrent Neural Networks in Keras, build with long short-term memory units (LSTMs), and to explore text prediction/generation. In this notebook I will focus on a character by character prediction and mainly follow [this blog](https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/). To understand the steps that I take, I often use the [Keras documentation](https://keras.io/getting-started/sequential-model-guide/) and try to provide sources as much as possible. 
# 
# To have an interesting story, I will try to generate 'The Eleventh Commandment' using a RNN based on Exodus - the second chapter of the King James Bible. At first I tried to use all Books of Moses, but this seems to be too much to run in a kernel. Anyway, thanks for reading this kernel and any feedback is welcome!

# In[ ]:


# Libraries
import numpy as np 
import pandas as pd
import re # re finditer (for finding string in text)
import nltk # natural language toolkit
from datetime import datetime 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import os
print(os.listdir("../input")) #pg10 


# In[ ]:


# load ascii text and covert to lowercase
raw_text = open("../input/pg10.txt").read()
raw_text = raw_text.lower()


# ## Data Preparation
# In the beginning of the book, there are some sentences on Project Gutenberg, it's license etc. The text starts after the second three asterices in a row, so the text before this will be removed. (Later on I also remove the first chapter Genesis, which makes the first lines of code redundant. The reason that I still provide this, is that it is easier in the future to change the script so that it includes the first chapter)

# In[ ]:


# Show text
raw_text[1:2000]


# In[ ]:


# Find index of ***
for match in re.finditer('[*][*][*]', raw_text[1:2000]):
    index_gut = match.start() + 4 # add 3 asterices plus 1

# Remove first sentences
raw_text = raw_text[index_gut:]


# Normally I would use the complete text now in raw_text (or a larger subset), but now a subset is used in order to speed up the training process. Most probably using only one chapter results in a less powerfull generative model, but this is always a trade-off (speed vs quality). This subset will consist only of one of the 5 Books of Moses: Exodus ( the others are Genesis, Leviticus, Numeri and Deuteronomium) - also the first chapters of the Bible. These are written by the same author (Moses) and might therefore be more related to each other than to the other chapters. In turn, it might be interesting to see what output the model gives when training on all Books of Moses. Either way, everything except Exodus will be dropped. This chapter contains the Ten Commandments (we will save the index of the Tenth Commandment), so if we're lucky, we might be able to generate an Eleventh Commandment.

# In[ ]:


fs = 'The Second Book of Moses'
for match in re.finditer(fs.lower(), raw_text):
    index_ch2 = match.start()

fs = 'The Third Book of Moses'
for match in re.finditer(fs.lower(), raw_text):
    index_ch3 = match.start()

# For the books of Moses (the first 5 chapters), use:
# fs = 'The Book of Joshua' 

# Keep Exodus:
raw_text = raw_text[index_ch2:index_ch3]

# Find the 10th Commandment
tc = 'Thou shalt not covet thy neighbour'
for match in re.finditer(tc.lower(), raw_text):
    index_TC = match.start()


# To feed the LSTM Recurrent NN, the text needs to be converted to integers. Here this will be done by mapping each unique character in the text to an integer. Note that punctuation marks are not removed, as these are important for proper text (sentence) generation. In the end a character-by-character generative model will be created, but we can do this similarly for words or sentences.

# In[ ]:


# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))


# These are the unique characters found in the text with their corresponding mappings.

# In[ ]:


print(char_to_int)


# The set now contains 177,124 characters, of which 47 unique characters (excluding capitalized characters). (The complete text has over 4 million characters, with 54 unique characters. The first five chapters contain 852,018 characters, of which 48 unique).

# In[ ]:


n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)


# Now the train set has to be defined. First I follow the example given in the link above in order to get used to the generative model. Here the text is arbitrarily split up in 'sentences' of 100 characters, which are used to predict the 101th character. This is done for every letter in the subset (except the first 100 characters of course). This results in 177,024 patterns (177,124 - the first 100 characters).

# In[ ]:


# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)


# Below are the first 3 patterns. Note that they are almost similar, which should be the case (as explainded above, all 100 characters are used to predict the next character. The second pattern is thus the first pattern minus the first character, plus the y of the first pattern).

# In[ ]:


print(dataX[1:4])


# A LSTM network expects the form [samples, time steps, features], so the list of imput sequences has to be transformed.

# In[ ]:


X = np.reshape(dataX, (n_patterns, seq_length, 1))


# Also, as the LSTM network uses the sigmoid activation function by default, it is easier to normalise the integers between 0 and 1.

# In[ ]:


X = X / float(n_vocab)


# In order to predict the probability of each of the different characters, each y value is converted using one hot encoding. This results in a sparce vector with the length of 54, with only a 1 representing the letter in the column for that letter.

# In[ ]:


y = np_utils.to_categorical(dataY)


# Then we can also already prepare a dictionary to translate the integers back to characters. This will be used later on.

# In[ ]:


int_to_char = dict((i, c) for i, c in enumerate(chars))


# ## LSTM Recurrent Neural Network
# 
# The problem with vanilla recurrent neural networks, constructed from regular neural network nodes, is that as we try to model dependencies between words or sequence values that are separated by a significant number of other words, we experience the vanishing gradient problem [(source)](http://adventuresinmachinelearning.com/keras-lstm-tutorial/). LSTMs help preserve the error that can be backpropagated through time and layers. By maintaining a more constant error, they allow recurrent nets to continue to learn over many time steps, thereby opening a channel to link causes and effects remotely. [(source)](https://deeplearning4j.org/lstm.html).
# 
# Now we are ready to create the LSTM model.
# 
# The LSTM model is a type of sequential model (a linear stack of layers), which we can define with Sequential(). We can simply add layers via the .add() method. The model needs to know what input shape it should expect and for this reason, the first layer in a Sequential model (and only the first, because following layers can do automatic shape inference) needs to receive information about its input shape [(source)](https://keras.io/getting-started/sequential-model-guide/). This we do by passing an input_shape argument to the first layer. Here we define a single hidden LSTM layer with 256 memory units (neurons) with the input shape is 100x1 (X.shape[1] by X.shape[2]).
# 
# A dropout on the input means that for a given probability, the data on the input connection to each LSTM block will be excluded from node activation and weight updates. In Keras, this is specified with a dropout argument when creating an LSTM layer. The dropout value is a percentage between 0 (no dropout) and 1 (no connection) [(source)](https://machinelearningmastery.com/use-dropout-lstm-networks-time-series-forecasting/). This can thus be seen as a regularization method. The dropout rate here is set at 20%.
# 
# Dense is a regular densely-connected NN layer, here used as the output layer. It uses the softmax activation function to output a probability prediction for each of the 47 characters.
# 
# In order to compile a Keras model, we need two arguments: the loss function and the optimizer. As our target is in a categorical format (each character represents a category and y is a 47-dimensional vector that is all-zeros except for a 1 at the index corresponding to the class of the sample), we should use categorical_crossentropy as the loss function. Following the example, I use the Adam optimizer as it is straightforward to implement, is computationally efficient and has little memory requirements [(source)](https://arxiv.org/abs/1412.6980v8).
# 
# Then we have constructed a basic LSTM model and another layer can be added with the same .add() method above. No input shape has to be defined. 

# In[ ]:


# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


# Because training this model is slow, checkpoints will be used to record all network weights. The set of weights with the lowest loss will be used in the generative model.

# In[ ]:


# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


# For performance, I only use 6 epochs. Recommended is to use way more (50 to 100). The algorithm will take the first 128 samples from X and train its network. Next it takes the second 128 samples (from the 129th) and train again. This is repeated until all samples are propagated through the network.

# In[ ]:


startTime = datetime.now()
model.fit(X, y, epochs=6, batch_size=128, callbacks=callbacks_list)
print('Time elapsed: ', datetime.now() - startTime)


#  As the number of epochs has to be increased anyway and the loss decreases at every epoch, we can simply select the last epoch for the lowest loss.

# In[ ]:


filenamels = [filename for filename in os.listdir('.') if filename.startswith('weights-improvement-06-')]
# type(filename) is a list, must be string
filename = ''.join(filenamels)
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')


# Now the model has been compiled with the lowest weights and we can check if we can generate some new text. Here I will start with the Tenth Commandment, predict the next character, update the sequence (that is, remove the first character of the Tenth Commandment and add the predicted character), predict the next character, etc.

# In[ ]:


start = index_TC
pattern = dataX[start]

print("The Tenth Commandment:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

for i in range(100):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    #print(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]

print("The Eleventh Commandment:")    
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")


# That's it.  A great insight retrieved with the power of Python! 
# 
# But seriously, this sentence doesn't make much sense as it seems to repeat the same sequence over and over. There are many ways to improve this model. Besides increasing the corpus to more or all chapters and the number of epochs, additional layers and additional nodes can be added to the model (this I will do shortly). Also instead of using the 'sentences' that are arbitrarily cut off at 100 characters, using padded sentences could improve this model.
