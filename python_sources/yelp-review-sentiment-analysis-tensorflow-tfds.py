#!/usr/bin/env python
# coding: utf-8

# In this notebook, we are going to perform a sentiment analysis on Yelp Reviews, which means we're going to train a model to "read" a review and determine if it's a positive or negative review.

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


# To get the data, we can download it directly using the Tensorflow Datasets (TFDS) package. TFDS has a wide variety of image, audio, text, and other kinds of datasets that can easily be donwloaded and used. For more information, visit: https://www.tensorflow.org/datasets/overview

# In[ ]:


get_ipython().system('pip install tensorflow-datasets')


# In[ ]:


import tensorflow as tf
import tensorflow_datasets as tfds


# In[ ]:


data = tfds.load('yelp_polarity_reviews', split='train', shuffle_files=True)


# The data comes in a tensor form, but I want to save the reviews as strings and the labels as integers (0 for negative, and 1 for positive).

# In[ ]:


reviews = []
polarity = []

for i in data.take(20000):
    reviews.append((i['text'].numpy().decode("utf-8")))
    polarity.append(int(i['label']))


# Let's take a look at a review.

# In[ ]:


reviews[3]


# Looking through, the reviews have some escape sequences ("\\n", etc) in the text. I'll remove these to make the analysis better.

# In[ ]:


def clean_text(review):
    cleaned = review.replace("\\n", " ")
    cleaned = cleaned.replace("\'", "'")
    cleaned = cleaned.replace("\\r", " ")
    cleaned = cleaned.replace("\\""", " ")
    return cleaned


# In[ ]:


reviews = [clean_text(review) for review in reviews]


# In[ ]:


reviews[3]


# That's better. Now, I'm going to use Keras preprocessing layers to tokenize the reviews (turn the text into an array of numbers). For more information about the Tokenizer, visit: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer

# In[ ]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000
max_length = 200

tokenizer = Tokenizer(num_words = vocab_size)
tokenizer.fit_on_texts(reviews)


# We have to make sure all of the input into our Tensorflow model is the same size, so we pad the sequences to all be the same length.

# In[ ]:


sequences = tokenizer.texts_to_sequences(reviews)
padded_sequences = pad_sequences(sequences, max_length, padding = 'post')


# In[ ]:


padded_sequences


# Now, we split our data into training, testing, and validation sets. For more information about each of these, visit: https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7

# In[ ]:


from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(padded_sequences, polarity)
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y)


# In[ ]:


len(train_x), len(train_y)


# Now, we build our Tensorflow model. For text data, we usually use an Embedding layer. I've also chosen to use bidirectional LSTM layers, but there are many different kinds of layers that you could use.

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Flatten

model = Sequential()
model.add(Embedding(vocab_size, 64, input_length=max_length-1))
model.add(Bidirectional(LSTM(20, return_sequences = True)))
model.add(Bidirectional(LSTM(20)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


model.fit(np.array(train_x), np.array(train_y), epochs = 3, verbose = 1, 
          validation_data = (np.array(val_x), np.array(val_y)))


# Let's evaluate our model on the test set:

# In[ ]:


print("Accuracy: ", model.evaluate(np.array(test_x), np.array(test_y))[1])


# Let's look at a review in the test set.

# In[ ]:


tokenizer.sequences_to_texts([test_x[0]])


# The review looks pretty positive. Let's see what the model thinks:

# In[ ]:


np.round(max(model.predict(test_x[0])))


# The model returned a 1, which means it identified it as positive. Yay!

# In[ ]:




