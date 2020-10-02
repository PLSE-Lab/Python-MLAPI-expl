#!/usr/bin/env python
# coding: utf-8

# ## About 
# This notebook contains a very fast and simple NLP example in Python.
# 
# This work is part of a series called [NLP in minutes - fast, simple examples](https://www.kaggle.com/jamiemorales/nlp-in-minutes-fast-simple-example)
# 
# The approach is designed to help grasp the applied artificial intelligence workflow in minutes. It is not an alternative to actually taking the time to learn. What it aims to do is help someone get started fast and gain intuitive understanding of the typical steps early on.

# ## Step 0: Understand the problem
# What we're trying to do here is to classify whether a news headline is sarcastic.

# ## Step 1: Set-up and understand data
# In this step, we layout the tools we will need to solve the problem identified in the previous step. We want to inspect our data sources and explore the data itself to gain an understanding of the data for preprocessing and modeling.

# In[ ]:


# Set-up libraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow import keras


# In[ ]:


# Check source
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Load data
df = pd.read_json('../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json', lines=True)
df.head()


# In[ ]:


# Look at some details
df.info()


# In[ ]:


# Look at breakdown of label
df['is_sarcastic'].value_counts()


# ## Step 2: Prepare data and understand some more
# In this step, we perform the necessary transformations on the data so that the neural network would be able to understand it. Real-world datasets are complex and messy. For our purposes, most of the datasets we work on in this series require minimal preparation.

# In[ ]:


# Split data into 80% training and 20% validation
sentences = df['headline']
labels = df['is_sarcastic']

train_sentences, val_sentences, train_labels, val_labels = train_test_split(sentences, labels, test_size=0.2, random_state=0)

print(train_sentences.shape)
print(val_sentences.shape)
print(train_labels.shape)
print(val_labels.shape)


# In[ ]:


# Tokenize and pad
vocab_size = 10000
oov_token = '<00V>'
max_length = 100
padding_type = 'post'
trunc_type = 'post'


tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

val_sequences = tokenizer.texts_to_sequences(val_sentences)
val_padded = pad_sequences(val_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)


# ## Step 3: Build, train, and evaluate neural network
# First, we design the neural network, e.g., sequence of layers and activation functions. 
# 
# Second, we train the neural network, we iteratively make a guess, calculate how accurate that guess is, and enhance our guess. The first guess is initialised with random values. The goodness or badness of the guess is measured with the loss function. The next guess is generated and enhanced by the optimizer function.
# 
# Lastly, we apply use the neural network on previously unseen data and evaluate the results.

# In[ ]:


# Build and train neural network
embedding_dim = 16
num_epochs = 10
batch_size = 100

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy']
             )

history = model.fit(train_padded, train_labels, batch_size=batch_size, epochs=num_epochs, 
                    verbose=2)


# In[ ]:


# Apply neural network
val_loss, val_accuracy = model.evaluate(val_padded, val_labels)
print('Val loss: {}, Val accuracy: {}'.format(val_loss, val_accuracy*100))

quick_test_sentence = [
    'canada is flattening the coronavirus curve',
    'canucks take home the cup',
    'safety meeting ends in accident'
    
]

quick_test_sequences = tokenizer.texts_to_sequences(quick_test_sentence)
quick_test_padded = pad_sequences(quick_test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
quick_test_sentiments = model.predict(quick_test_padded)

for i in range(len(quick_test_sentiments)):
    print('\n' + quick_test_sentence[i])
    if 0 < quick_test_sentiments[i] < .50:
        print('Unlikely sarcasm. Sarcasm score: {}'.format(quick_test_sentiments[i]*100))
    elif .50 < quick_test_sentiments[i] < .75:
        print('Possible sarcasm. Sarcasm score: {}'.format(quick_test_sentiments[i]*100))
    elif .75 >  quick_test_sentiments[i] < 1:
        print('Sarcasm. Sarcasm score:  {}'.format(quick_test_sentiments[i]*100))
    else:
        print('Not in range')


# ## Learn more
# If you found this example interesting, you may also want to check out:
# 
# * [Deep learning - very fast fundamental examples](https://www.kaggle.com/jamiemorales/deep-learning-very-fast-simple-examples)
# * [Machine learning in minutes - very fast fundamental examples in Python](https://www.kaggle.com/jamiemorales/machine-learning-in-minutes-very-fast-examples)
# * [List of machine learning methods & datasets](https://www.kaggle.com/jamiemorales/list-of-machine-learning-methods-datasets)
# 
# Thanks for reading. Don't forget to upvote.
