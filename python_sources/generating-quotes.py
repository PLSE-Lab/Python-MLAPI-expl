#!/usr/bin/env python
# coding: utf-8

# # **Generating Quotes using LSTM**

# In this kernel, I will walk you through the process of generating text using LSTM. For purpose of this tutorial, we will use Quotes dataset and train our model to create our custom quotes generator.
# 
# Let's start by importing necessary libraries and loading in the dataset.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import layers
from keras.models import Sequential
import keras.utils as ku
from keras.callbacks import EarlyStopping


# In[ ]:


# Loading the dataset
data = pd.read_json('/kaggle/input/quotes-dataset/quotes.json')
print(data.shape)
data.head()


# The dataset has features such as:
# * Quote
# * Author
# * Tags
# * Popularity
# * Category
# 
# But for our task at hand, we are interested in only the Quote feature of the dataset. If you look at the Quote column, a single quote is attributed to multiple categories such as life, happiness, etc. So, we will drop the duplicate quotes and consider only unique quotes.

# In[ ]:


# Dropping duplicates and creating a list containing all the quotes
quotes = data['Quote'].drop_duplicates()
print(f"Total Unique Quotes: {quotes.shape}")

# Considering only top 3000 quotes
quotes_filt = quotes.sample(3000)
print(f"Filtered Quotes: {quotes_filt.shape}")
all_quotes = list(quotes_filt)
all_quotes[:2]


# Next step is to preprocess and prepare the data for a Text Generation model. 
# 
# First, we will tokenize the text usign Keras **Tokenizer** class to create a vocabulary and convert the text into sequence of token indexes.
# 
# There are two levels at which you can generate text:
# 1. Character level
# 2. Word level
# 
# Here, first we will focus on word level text generation. Suppose, let's consider the quote **"Don't cry because it's over, smile because it happened"**, we have to prepare our data in the format below:
# ![image.png](attachment:image.png)
# The reason why we need to prepare our data in such a way is very intuitive because, even when we write any piece of text, we will form sentences word by word i.e., the next word we write depends upon the previous words we have used. So, when we give the model sequences in this format, it will also try to learn the sequence and predict the next possible word exactly how we do.

# In[ ]:


# Tokeinization
tokenizer = Tokenizer()

# Function to create the sequences
def generate_sequences(corpus):
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    print(f"Total unique words in the text corpus: {total_words}")
    input_sequences = []
    for line in corpus:
        seq = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(seq)):
            ngram_seq = seq[:i+1]
            input_sequences.append(ngram_seq)
            
    return input_sequences, total_words

# Generating sequences
input_sequences, total_words = generate_sequences(all_quotes)
input_sequences[:5]


# Now that we have the data in required format, but each sequences are of different length. So, before feeding into the model, we will first pad the sequences to same length.
# 
# Also, we need to create predictor and label from the prepared sequences by taking all the tokens except the last one as predictors and the last token as label (For example, think of it like the data in the above table: "Don't cry" as predictors and "because" as label).

# In[ ]:


# Generating predictors and labels from the padded sequences
def generate_input_sequence(input_sequences):
    maxlen = max([len(x) for x in input_sequences])
    input_sequences = pad_sequences(input_sequences, maxlen=maxlen)
    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, maxlen

predictors, label, maxlen = generate_input_sequence(input_sequences)
predictors[:1], label[:1]


# Finally, we are done with the preprocessing part of task. Now, we will start building our LSTM model for text generation. You can think of this model as a multiclass text classification task- given the previous words, the model will predict the next word which has high probability.
# 
# **Model Architecture:**
# * Embedding layer with the embedding dimension of 64
# * LSTM Layer with 128 units with dropout
# * A dense layer with number of units equal to the total words in the vocabulary with **softmax** activation since it is a mulitclass classification task.
# * The optimizer we use here is **Adam**, loss is **categorical_crossentropy**, and an epoch of 50.

# In[ ]:


# Building the model
embedding_dim = 64

def create_model(maxlen, embedding_dim, total_words):
    model = Sequential()
    model.add(layers.Embedding(total_words, embedding_dim, input_length = maxlen))
    model.add(layers.LSTM(128, dropout=0.2))
    model.add(layers.Dense(total_words, activation='softmax'))
    
    # compiling the model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

model = create_model(maxlen, embedding_dim, total_words)
model.summary()


# In[ ]:


predictors.shape , label.shape, maxlen


# In[ ]:


# Training the model
# model.fit(predictors, label, epochs=50, batch_size=64)


# The model has been trained for almost two hours for only 50 epochs. So, will save the model to avoid training every time we want to generate a pice of text.

# In[ ]:


# Save the model for later use
# model.save("Quotes_generator.h5")


# In[ ]:


# Loading the model
from keras.models import load_model

Quotes_gen = load_model("../input/quote-generator-trained-model/Quotes_generator.h5")


# In[ ]:


Quotes_gen.summary()


# Now that we have our trained model, we will create a function to generate text.
# 
# The function takes in the trained model, the input words (also called seed text), how many words to genereate and maximum squence length. The function then tokenize the text, padds it and predict using our trained model.
# 
# The model predicts one word at a time. So after every prediction, we will get the word for the predicted label and append it to the seed_text. This process continues for the specified number of words you want to genereate. And once it is done, the text will then be returned.
# 

# In[ ]:


# Text generating function
def generate_quote(seed_text, num_words, model, maxlen):
    
    for _ in range(num_words):
        tokens = tokenizer.texts_to_sequences([seed_text])[0]
        tokens = pad_sequences([tokens], maxlen=maxlen, padding='pre')
        
        predicted = model.predict_classes(tokens)
        
        output_word = ''
        
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text = seed_text + " " + output_word
    
    return seed_text


# In[ ]:


# Let's try to generate some quotes
print(generate_quote("Passion", num_words = 10, model= Quotes_gen, maxlen=maxlen))


# In[ ]:


print(generate_quote("Love", num_words = 20, model= Quotes_gen, maxlen=maxlen))


# In[ ]:


print(generate_quote("legend", num_words = 15, model= Quotes_gen, maxlen=maxlen))


# In[ ]:


print(generate_quote("consistency matters", num_words = 15, model= Quotes_gen, maxlen=maxlen))


# In[ ]:


print(generate_quote("Follow your passion", num_words = 20, model= Quotes_gen, maxlen=maxlen))


# The generated quotes looks okayish but still can be improved a lot. Due to my system specs, I had train it using only 3000 quotes and 50 epochs with only one LSTM layer. You can tune these parameters and train for more epochs to get higher quality results.
# 
# This ends our task of generating text using LSTM. Here, I have used word level text generation but this works only if you have huge amount of data. Even, if you have huge amount of data, you will face huge dimensionality for the label, since you will be using total words in the dictionary for prediction and this leads to system crash, if you are training in a relatively low power system like mine.
# 
# In such cases, you can use character level text generation since total characters in english is only 26 and adding up some punctuations would take this to max 30-40 characters. In, the follow up versions of this notebook, I will train a character level model and compare its generation quality with the word level model.
# 
# Till then, Happy Learning!!

# **Reference:** https://medium.com/@shivambansal36/language-modelling-text-generation-using-lstms-deep-learning-for-nlp-ed36b224b275 

# In[ ]:




