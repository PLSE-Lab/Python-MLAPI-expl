#!/usr/bin/env python
# coding: utf-8

# **** It is to be told in advance that this program is not meant to be a working model, it is just an attempt to explain how can we use the similar process without the involvment of much complications to achieve the desired results****
# In order to perform text processing we need various tools to create word embeddings, for that we are going to use Tensorflow and Keras.
# Also we will be using keras preprocessing tools for operations like padding and sequence generation as well.

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
import numpy as np


# Let's now create a test toy dataset of our own and perform preprocessing on it. In order to feed our training data in our Neural Network we need to convert our data to word vectors of finite length so that our network can learn from the data from here onwards.
# In order to create the word vectors from sentences we need to tokenize the words so that we can treat each word as a seperate entity.
# Also, for giving the input of finite length we need to use a vector of fixed size, but we also know that each sentence is not going to be of same size that's why we need to pad the sequence to make it of a finite length.

# In[ ]:


sentences = [
    'I love my cat',
    'You love my dog',
    'i dont like coffee',
    'my mother is very beautiful',
    'i hate it when people touch my shoulder',
    'i adore my brother and sister',
    'I love my dog',
    'i dont like milk cream',
    'folwers are very nice',
    'i was hurt when i did not qualify for the exams',
    'i dont like exams',
    'i adore small puppies',
    'i dont like mosquitoes',
    'i love small talks',
    'i get offended cery easily',
    'i like readiing books',
    'i get angry when someone disrespects me',
    'i appreciate your tough muscles',
    'i hate that you love her',
    'you cant see me because i am bad',
    'i have a crush on you'
    
]


output = [1,1,0,1,0,1,1,0,1,0,0,1,0,1,0,1,0,1,0,0,1]
output = np.array(output).reshape(-1,1)
tokenizer = Tokenizer(num_words =100,oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences,15)
print(output.shape,padded.shape)
print(output[:4])
print(padded[:4])


# The network that we are going to use is a Sequential model, with Bidirectional LSTM layer as one of the layer. The first layer is going to be the word embedding layer where words are stored in a multidimensional space in such a way that the words which are related appear close to each other after training.
# The output is going to be binary that's we are going to use binary crossentropy loss, with adam optimizer.

# In[ ]:


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(100,5, input_length = 15),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(9)),
    tf.keras.layers.Dense(8, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# In[ ]:


model.fit(padded, output,epochs=55, batch_size=8, verbose=2)
# Final evaluation of the model
#scores = model.evaluate(X_test, y_test, verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:


sentences1 = [
    'fast and furious is a very good movies',
    'I love my cat movie',
    'i dont like bhoot',
    'i appreciate your hard work',
    'you dont like me'
]
output1 = [1,1,0,1,0]
output1 = np.array(output).reshape(-1,1)
sequences1 = tokenizer.texts_to_sequences(sentences1)
padded1 = pad_sequences(sequences1,15)
model.predict(padded1)
model.evaluate(padded1,output1)


# In[ ]:


sentences1 = [
    'fast and furious is a very good movies',
    'I love my cat movie',
    'i dont like bhoot',
    'i appreciate your hard work',
    'you dont like me',
    'i hate that you love me',
    'you see me because i am good '
]
sequences1 = tokenizer.texts_to_sequences(sentences1)
padded1 = pad_sequences(sequences1,15)
model.predict(padded1)


# In[ ]:




