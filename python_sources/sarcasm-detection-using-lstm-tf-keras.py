#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# As next step we will load the file using pandas asnd filter necessery columns

# In[ ]:


file_path1 = '/kaggle/input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json'
df = pd.read_json(file_path1,lines=True)
df = df[['headline','is_sarcastic']]
df.head()


# In[ ]:


headlines = df['headline'].values.tolist()
sarcastic = df['is_sarcastic'].values.tolist()

print('Length of data {}'.format(len(headlines)))


# Now we will split the data to train and test

# In[ ]:


training_size = 20000
test_size = 6709

train_x = headlines[:training_size]
test_x = headlines[training_size:]
train_y = np.array(sarcastic[:training_size])
test_y = np.array(sarcastic[training_size:])


# In[ ]:


print(train_x[0])
print(train_y[0])


# Now we will follow the following steps: <br>
# 1. Tokenize the sentances.
# 2. Convert it to sequence.
# 3. Padding the sequences.

# In[ ]:


# vocab_size = 2000   #number of words in tokenizer
embedding_dim = 100
max_len = 16

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(train_x)

word_index = tokenizer.word_index
vocab_size = len(word_index)
sequence_train = tokenizer.texts_to_sequences(train_x)
seq_padd_train = pad_sequences(sequence_train,padding='post',truncating='post',maxlen=max_len)

#test
sequence_test = tokenizer.texts_to_sequences(test_x)
seq_padd_test = pad_sequences(sequence_test,padding='post',truncating='post',maxlen=max_len)


# In[ ]:


print(sequence_train[0])
print(seq_padd_train[0])
print(seq_padd_train.shape)


# Now we will use transfer learning for creating an embedding matrix.<br>
# 1. Download pretrained glove file
# 2. Create embedding dict
# 3. Create embedding matrix

# In[ ]:


get_ipython().system('wget --no-check-certificate     https://storage.googleapis.com/laurencemoroney-blog.appspot.com/glove.6B.100d.txt     -O /tmp/glove.6B.100d.txt')


# In[ ]:


embeddings_index = {};
with open('/tmp/glove.6B.100d.txt') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:],dtype='float32')
        embeddings_index[word] = coefs
    


# In[ ]:


# creating embedding matrix
embeddings_matrix = np.zeros((vocab_size+1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector


# In[ ]:


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_len,weights=[embeddings_matrix], trainable=False),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True)),
    #return_sequences: will ensure output of first LSTM layer matches next
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


# In[ ]:


num_epochs = 20
history = model.fit(seq_padd_train, train_y , epochs=num_epochs, validation_data=(seq_padd_test,test_y))


# In[ ]:


train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(num_epochs)

plt.subplot(2,1,1)
plt.plot(epochs,train_accuracy)
plt.plot(epochs,val_accuracy)
plt.legend(['train_acc','val_acc'])
plt.title('Accuracy')
plt.show()

plt.subplot(2,1,2)
plt.plot(epochs,train_loss)
plt.plot(epochs,val_loss)
plt.legend(['train_loss','val_loss'])
plt.title('Loss')
plt.show()


# The model is overfitting.You can try out various methods to overcome it.
# 
