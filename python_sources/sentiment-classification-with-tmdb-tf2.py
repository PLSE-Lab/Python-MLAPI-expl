#!/usr/bin/env python
# coding: utf-8

# ## **1. Background**

# ![Natural language processing](https://landbot.io/wp-content/uploads/2019/11/natural-language-processing-chatbot.jpg)

# **What is Natural Language Processing?**
# 
# From wikipedia, Natural language processing (NLP) is a subfield of linguistics, computer science, information engineering, and artificial intelligence concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data.
# 
# **What is Sentiment Classification?**
# 
# Sentiment analysis (also known as opinion mining or emotion AI) refers to the use of natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study affective states and subjective information. Sentiment analysis is widely applied to voice of the customer materials such as reviews and survey responses, online and social media, and healthcare materials for applications that range from marketing to customer service to clinical medicine.
# 
# **What is Tokenizer?**
# 
# Tokenization is a necessary first step in many natural language processing tasks, such as word counting, parsing, spell checking, corpus generation, and statistical analysis of text.
# 
# Tokenizer is a compact pure-Python (2 and 3) executable program and module for tokenizing Icelandic text. It converts input text to streams of tokens, where each token is a separate word, punctuation sign, number/amount, date, e-mail, URL/URI, etc. It also segments the token stream into sentences, considering corner cases such as abbreviations and dates in the middle of sentences.[Tokenizer](https://pypi.org/project/tokenizer/)
# 
# **What is Padding?**
# 
# As a same approach in Convolution Neural Network, Padding assure the input layer have the same shape for the model. 
# 
# **What is LSTM (long short term memory)?**
# 
# Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture used in the field of deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections. It can not only process single data points (such as images), but also entire sequences of data (such as speech or video). For example, LSTM is applicable to tasks such as unsegmented, connected handwriting recognition, speech recognition and anomaly detection in network traffic or IDS's (intrusion detection systems)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## **2. Data exploratory analysis**

# ### **2.1 Data overview**

# ![IMDB 50 review datasets](https://o.aolcdn.com/images/dims?quality=85&image_uri=https%3A%2F%2Fo.aolcdn.com%2Fimages%2Fdims%3Fcrop%3D908%252C537%252C0%252C0%26quality%3D85%26format%3Djpg%26resize%3D1600%252C947%26image_uri%3Dhttps%253A%252F%252Fs.yimg.com%252Fos%252Fcreatr-uploaded-images%252F2019-08%252F560e5d20-c833-11e9-bf26-36635805fe83%26client%3Da1acac3e1b3290917d92%26signature%3D639a4965c41ca6cec13652498f65cfc97170ea5d&client=amp-blogside-v2&signature=765e155477177a69b93eac5611145d4241be6071)

# This dataset contains movie reviews along with their associated binary sentiment polarity labels. It is intended to serve as a benchmark for sentiment classification. This document outlines how the dataset was gathered, and how to use the files provided.
# 
# **Dataset**
# 
# The core dataset contains 50,000 reviews. The overall distribution of labels is balanced (25k pos and 25k neg). We also include an additional 50,000 unlabeled documents for unsupervised learning.
# 
# In the entire collection, no more than 30 reviews are allowed for any given movie because reviews for the same movie tend to have correlated ratings. Further, the train and test sets contain a disjoint set of movies, so no significant performance is obtained by memorizing movie-unique terms and their associated with observed labels. In the labeled train/test sets, a negative review has a score <= 4 out of 10, and a positive review has a score >= 7 out of 10. Thus reviews with more neutral ratings are not included in the train/test sets. In the unsupervised set, reviews of any rating are included and there are an even number of reviews > 5 and <= 5.

# ### **2.2 Data pre-processing**

# The first step is to load the data to global environment.

# In[ ]:


df = pd.read_csv("/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")


# In[ ]:


df.head()


# To visualize the work and choose the proper approach to pre-process data and visualize the text data, here come some task as my defined.
# 
# 1) Data shape. From this to split our data to train and test data.
# 
# 2) What is the most common words? Use different approach to visualize the most common words. From this we can find out if any inappropriate words which could be removed.
# 
# 3) What is the distribution of the sentences length? From this we can choose the proper max_length of sentence.
# 
# 4) How many total words from our dataset? From this we can choose the vocabulary size for our model.

# In[ ]:


from collections import Counter
Counter(" ".join(df["review"]).lower().split()).most_common(100)


# We could see some abnormal words such as <br /><br />, then we should replace them by a null or space value.

# In[ ]:


#import string as str
#df['review'] = [i.replace('<br>', '').str.replace('</br>', '') for i in df['review']]
df['review'] = df['review'].str.replace('<br />','')
df['review'] = df['review'].str.lower()


# In[ ]:


plt.figure()
plt.hist(df['review'].str.split().apply(len).value_counts())
plt.xlabel('number of words in sentence')
plt.ylabel('frequency')
plt.title('Words occurrence frequency')


# In[ ]:


print('The maximum length of a sentence is: ',np.max(df['review'].str.split().apply(len).value_counts()))
print('The average lenth of a sentence is: ', np.average(df['review'].str.split().apply(len).value_counts()))


# In[ ]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df['sentiment'] = label_encoder.fit_transform(df['sentiment'])
df.head()


# In[ ]:


sentences = np.array(df['review'])
labels = np.array(df['sentiment'])


# Split data to train and test for modeling and performance evaluation.

# In[ ]:


training_sentences, testing_sentences,training_labels, testing_labels = train_test_split(sentences, labels, test_size = 0.2)


# ## **3. Modeling**

# In[ ]:


# choose hyper parameters to tune
vocab_size = 20000 #(before 10000)
embedding_dim = 150 #(before 16)
max_length =  400 #(was 32)
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"


# import Tokenizer & fit on training test
tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

# convert text to sequences
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen = max_length,
                                padding = padding_type,
                                truncating = trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen = max_length,
                                padding = padding_type,
                                truncating = trunc_type)

    # modeling
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    
    # option 1: Flatten
    #tf.keras.layers.Flatten(),
    #tf.keras.layers.GlobalAveragePooling1D(),
    
    # option 2: LSTM
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    #tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.15),

    # option 3: GRU
    #tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64)),
    
    # option 4: Conv1D
    #tf.keras.layers.Conv1D(128,5,activation='relu'),
    #tf.keras.layers.GlobalAveragePooling1D(),
    
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

model.summary()


# In[ ]:


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler

# compile model
model.compile(loss = 'binary_crossentropy',
            optimizer = Adam(learning_rate=0.001),
            metrics = ['accuracy'])

# add early stopping
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# learning rate decay
def lr_decay(epoch, initial_learningrate = 0.001):#lrv
    return initial_learningrate * 0.9 ** epoch

# training model
num_epochs = 10
history = model.fit(training_padded, training_labels,
                    epochs=num_epochs,
                    callbacks=[LearningRateScheduler(lr_decay),
                              callback],
                    batch_size = 512,
                    validation_data = (testing_padded, testing_labels),
                    verbose=1)


# In[ ]:


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.title(print('vocab_size: ',vocab_size))
    plt.legend([string, 'val_' + string])
    plt.show()
    
plot_graphs(history, "accuracy")
plot_graphs(history,"loss")

