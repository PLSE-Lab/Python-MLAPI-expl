#!/usr/bin/env python
# coding: utf-8

# # Poetry with RNN
# 
# Today, we will generate poems. I love reading and writing poetry, so I thought it would be fun to see what a machine can come up with rather than my own mind. I'm using the [Gutenberg Poetry Corpus by Allison Parris](https://github.com/aparrish/gutenberg-poetry-corpus).
# 
# This is mostly a learning exercise and practice document for me. Hope you enjoy the read.
# 
# #### Recurrent Neural Network (RNN)
# A recurrent neural network is a network with loops in it, which allows previous outputs to be used as inputs to the current step while having a hidden state. The hidden state remembers some information about a sequence. It can be understood as multiple copies of the same network that each pass a message to the next network.
# 
# For more details, [visit this Medium article by Suvro Banerjee](https://medium.com/explore-artificial-intelligence/an-introduction-to-recurrent-neural-networks-72c97bf0912).
# 
# #### Long Short-Term Memory (LSTM) Networks
# RNNs fail to understand the context behind input because they only remember things for short durations of time. Therefore once a lot of words are fed in, information gets lost and predictions are less reliable. This is because of the Vanishing Gradient problem: as more layers using the sigmoid activation function (or others) are added to neural networks, the gradients of the loss function approach 0. It becomes difficult to train layers. Read more about it [on Towards Data Science](https://towardsdatascience.com/the-vanishing-gradient-problem-69bf08b15484).
# 
# We can use LSTM Networks, a version of a RNN, to fix this issue. With LSTMs, information flows through cell states, allowing for slight modification of information -- LSTMs can therefore selectively remember or forget things.

# ## Prepare Data

# In[ ]:


import json
import random
import numpy as np
import pandas as pd

all_lines = []
for line in open("../input/gutenberg-poetry-v001.ndjson"):
    all_lines.append(json.loads(line.strip()))


# In[ ]:


corpus = "\n".join([line['s'] for line in random.sample(all_lines, 1000)])


# ## Aside: Markov chain text generation
# 
# Markov chains generate sentences based on recombination of elements of known sentences: it analyzes the words and the probability of occurence of two consecutive words. The generation is randomized and based on the probability of each words.
# 
# Let's generate a poem with this method, and then continue with our RNN method to see how they compare.

# In[ ]:


import markovify


# In[ ]:


model = markovify.NewlineText(corpus)


# In[ ]:


for i in range(5):
    print()
    for i in range(random.randrange(1, 4)):
        print(model.make_short_sentence(30))


# ## Character-Level LSTM Text Generation
# 
# #### Import Libraries

# In[ ]:


from keras.layers import LSTM, Dense, Dropout, Flatten
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import np_utils


# #### Prepare Text

# In[ ]:


# Lowercase all text
text = corpus.lower()

chars = list(set(text))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

vocab_size = len(chars)
print('Vocabulary size: {}'.format(vocab_size))


# #### Prepare Data

# In[ ]:


# Data preparation
X = [] # training array
Y = [] # target array

length = len(text)
seq_length = 100 # number of characters to consider before predicting a character

# Iterate over length of text and create sequences stored in X, true values stored in Y
# true values being which character would actually come after sequence stored in X
for i in range(0, length - seq_length, 1):
    sequence = text[i:i + seq_length]
    label = text[i + seq_length]
    X.append([char_indices[char] for char in sequence])
    Y.append(char_indices[label])

print('Number of sequences: {}'.format(len(X)))


# In[ ]:


# Reshape dimensions
X_new = np.reshape(X, (len(X), seq_length, 1))
# Scale values
X_new = X_new/float(len(chars))
# One-hot encode Y to remove ordinal relationships
Y_new = np_utils.to_categorical(Y)

X_new.shape, Y_new.shape


# #### Create Model

# In[ ]:


model = Sequential()
# Add LSTM layer to compute output using 150 LSTM units
model.add(LSTM(150, input_shape = (X_new.shape[1], X_new.shape[2]), return_sequences = True))

# Add regularization layer to prevent overfitting.
# Dropout ignores randomly selected neurons during training ("dropped out").
# Ultimately, network becomes less sensitive to specific weights of neurons --> network is better at generalization.
model.add(Dropout(0.1))

model.add(Flatten())
# Dense layer with softmax activation function to approximate probability distribution of next best word
model.add(Dense(Y_new.shape[1], activation = 'softmax'))

# Compile model to configure learning process
# Categorical crossentropy: an example can only belong to one class
# Adam optimization algorithm updates a learning rate for each network weight iteratively as learning unfolds
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

# Use 1 epoch for sake of computational time
model.fit(X_new, Y_new, epochs = 1, verbose = 1)


# #### Generate Text

# In[ ]:


# Random start
start = np.random.randint(0, len(X)-1)
string_mapped = list(X[start])
full_string = [indices_char[value] for value in string_mapped]

# Generate text
for i in range(400):
    x = np.reshape(string_mapped, (1, len(string_mapped), 1))
    x = x / float(len(chars))
    
    pred_index = np.argmax(model.predict(x, verbose = 0))
    seq = [indices_char[value] for value in string_mapped]
    full_string.append(indices_char[pred_index])
    
    string_mapped.append(pred_index)
    string_mapped = string_mapped[1:len(string_mapped)]
    
# Combine text
newtext = ''
for char in full_string:
    newtext = newtext + char

print(newtext)


# ## Predicting Next Words

# In[ ]:


import keras.utils as ku
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding

# Lowercase all text
text = corpus.lower()
text = text.split('\n')

# Create Tokenizer object to convert words to sequences of integers
tokenizer = Tokenizer(num_words = None, filters = '#$%&()*+-<=>@[\\]^_`{|}~\t\n', lower = False)

# Train tokenizer to the texts
tokenizer.fit_on_texts(text)
total_words = len(tokenizer.word_index) + 1

# Convert list of strings into flat dataset of sequences of tokens
sequences = []
for line in text:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        sequences.append(n_gram_sequence)

# Pad sequences to ensure equal lengths
max_seq_len = max([len(x) for x in sequences])
sequences = np.array(pad_sequences(sequences, maxlen = max_seq_len, padding = 'pre'))

# Create n-grams sequence predictors and labels
predictors, label = sequences[:, :-1], sequences[:, -1]
label = ku.to_categorical(label, num_classes = total_words)


# #### Create Model

# In[ ]:


# Input layer takes sequence of words as input
input_len = max_seq_len - 1
model = Sequential()
model.add(Embedding(total_words, 10, input_length = input_len))
model.add(LSTM(150))
model.add(Dropout(0.1))
model.add(Dense(total_words, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

# Use 100 epoch for efficacy
model.fit(predictors, label, epochs = 100, verbose = 1)


# In[ ]:


# Function to generate line
def generate_line(text, next_words, max_seq_len, model):
    for j in range(next_words):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list], maxlen = max_seq_len - 1, padding = 'pre')
        predicted = model.predict_classes(token_list, verbose = 0)
        
        output_word = ''
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        text += ' ' + output_word
    return text


# In[ ]:


generate_line("gone were the", 5, max_seq_len, model)


# In[ ]:


generate_line("oh what sweet", 5, max_seq_len, model)


# ##### Thank you!
