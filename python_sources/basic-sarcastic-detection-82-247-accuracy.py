#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import json
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import numpy as np
df = pd.read_json("../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json", lines=True)
df.head()


# ****Importing Tensorflow API's and the data set of course

# In[ ]:


sentences = []
labels = []

vocab_size = 1000
embedding_dim = 16
max_length = 16
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 20000


# Declaring all the essential parameters that we are going to use in the following code

# In[ ]:


for index , rows in df.iterrows():
    my_list = rows.headline
    my_labels = rows.is_sarcastic
    
    sentences.append(my_list)
    labels.append(my_labels)
    

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]

training_labels = labels[0:training_size]
testing_labels = labels[training_size:]


# Putting all the headlines and Lables to one list and dividing them into testing and training sets respectively

# In[ ]:


tokenizer = Tokenizer(num_words = vocab_size ,oov_token = "<OOV>")
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen = max_length,padding = padding_type,truncating = trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded= pad_sequences(testing_sequences, maxlen = max_length,padding = padding_type,truncating = trunc_type)

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)


# Creating a Tokenizer object and fitting it onto sentences and Converting each sentence into a sequence of words using texts_to_sequences method. After that we also use padding to ensure all sentences are of same length.

# In[ ]:


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
    
    
    
    
    
])

model.compile(loss = 'binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
time.sleep(0.1)


# In[ ]:


num_epochs = 29


history = model.fit(training_padded,training_labels,epochs = num_epochs,validation_data = (testing_padded,testing_labels),verbose = 1)

result = model.evaluate(testing_padded,testing_labels)
print(result)


# Here we can see the accuracy is **82.24773%**

# In[ ]:



def plot_graphs(history,string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string,'val_'+string])
    plt.show()
plot_graphs(history,"accuracy")
plot_graphs(history,"loss")


# Loss function is tweaked but accuracy is not that high.
