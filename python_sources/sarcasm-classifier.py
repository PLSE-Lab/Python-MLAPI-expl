#!/usr/bin/env python
# coding: utf-8

# **NLP Model with TensorFlow**

# In[ ]:


# Run this to ensure TensorFlow 2.x is used
try:
  # %tensorflow_version only exists in Colab.
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
  pass


# In[ ]:


import json
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[ ]:


vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000


# **Features & Label**
# 
# From the download json data we take feature and label for training the NLP Classifier Model
# 
# Feature -> headline
# 
# Label   -> is_sarcastic

# In[ ]:


#Snippet is used to get the path 
import os
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


"""import pandas as pd
path = "../input/Sarcasm_Headlines_Dataset.json"
datastore = pd.read_json(path, lines=True)"""



data = [json.loads(line) for line in open('../input/sarcasm-headlines-dataset/Sarcasm_Headlines_Dataset.json', 'r')]

sentences = []
labels = []

for item in data:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])


# In[ ]:


training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]


# **NLP Steps**
# 
# By understanding the data we will classify the given data as sarcastic or not!
# 
# But to do that the general steps are need to be followed
# 
# **Tokenization**
# 
# This process is the first step where the given words will be converted to different id or word_index.
# Example -- I love ML can have a following word_index [0,1,2]
# 
# **Padding Sequences**
# 
# The padding sequence is used to have a padding value to the given sentence to make every sequence of sentence in a same length,since the neural net needs a fixed size of input.

# In[ ]:


tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)


# In[ ]:


# Need this block to get it to work with TensorFlow 2.x
import numpy as np
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)


# **Model Creation**
# 
# But you might be wondering at this point,we've turned our sentences into numbers,
# with the numbers being tokens representing words.
# But how do we get meaning from that?
# How do we determine if something is sarcastic just
# from the numbers?
# Well, here's where the context of embeddings come in.
# 
# Let's consider the most basic of sentiments.
# Something is good or something is bad.
# We often see these as being opposites,
# so we can plot them as having opposite directions.
# So then what happens with a word like "meh"?
# It's not particularly good, and it's not particularly bad.
# Probably a little more bad than good.
# Or the phrase, "not bad," which is usually
# meant to plot something as having
# a little bit of goodness, but not necessarily very good.
# 
# 
# 

# In[ ]:


from IPython.display import Image
source = "../input/imageda/Capture.JPG"
Image(source)


# Now, if we plot this on an x- and y-axis as shown in above image,
# 
# we can start to determine the good or bad sentiment
# as coordinates in the x and y.
# Good is 1, 0.
# Meh is minus 0.4, 0.7, et cetera.
# By looking at the direction of the vector,
# we can start to determine the meaning of the word.
# So what if you extend that into multiple dimensions instead
# of just two?
# What if words that are labeled with sentiments,
# like sarcastic and not sarcastic,
# are plotted in these multiple dimensions?
# And then, as we train, we try to learn
# what the direction in these multi-dimensional spaces
# should look like.
# Words that only appear in the sarcastic sentences
# will have a strong component in the sarcastic direction,
# and others will have one in the not-sarcastic direction.
# As we load more and more sentences
# into the network for training, these directions can change.
# And when we have a fully trained network
# and give it a set of words, it could look up
# the vectors for these words, sum them up, and thus, give us
# an idea for the sentiment.
# This concept is known as embedding.

# In[ ]:


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


num_epochs = 30
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)


# **Visualization **
# 
# Here we use the graphs to show the relationship between the accuracy and loss of both training and validation data.

# In[ ]:


import matplotlib.pyplot as plt


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")


# In[ ]:


reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_sentence(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_sentence(training_padded[0]))
print(training_sentences[2])
print(labels[2])


# In[ ]:


e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)


# **Prediction**
# 
# Here we specifiy the sentence in sarcastic and opposite way and the prediction gave show the result correctly

# In[ ]:


sentence = ["granny starting to fear spiders in the garden might be real", "game of thrones season finale showing this sunday night"]
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(model.predict(padded))

