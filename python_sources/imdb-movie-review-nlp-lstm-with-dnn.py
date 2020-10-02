#!/usr/bin/env python
# coding: utf-8

# # Importing necessary libraries
# 

# In[ ]:


import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import ModelCheckpoint


# # Visualizing Data
# As we can see our data is distributed evenly 25k positive reviews and 25k negative reviews count plot is shown in the figure.

# In[ ]:


dataset = pd.read_csv('../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')
negative = len(dataset[dataset['sentiment']=='positive'])
positive = len(dataset) - negative
sns.countplot(dataset['sentiment'])
print('Positive reviews are {} and negative reviews are {} of total {} '.format(positive,negative,len(dataset)))


# Converting the labels positve and negative as 1,0 so that they can be fed to the neural network to predict whether the given review is a positive or negative. Splitting of data 80% for the training and remaining 20% for testing.

# In[ ]:


le = LabelEncoder()
training_reviews,testing_reviews,training_labels,testing_labels  = train_test_split(dataset['review'].values,dataset['sentiment'].values,test_size = 0.2)
training_labels = le.fit_transform(training_labels)
testing_labels = le.fit_transform(testing_labels)


# # Pre-Processing The Text
# Using tokenizer to produce token for a given word and taking maximum length of 200 character of a review and after we simply truncate the input review and then padded the input to max len of 200. 

# In[ ]:


tokenizer = Tokenizer(num_words=10000,oov_token='<OOV>')
tokenizer.fit_on_texts(training_reviews)
word_index = tokenizer.word_index
training_sequence = tokenizer.texts_to_sequences(training_reviews)
testing_sequence = tokenizer.texts_to_sequences(testing_reviews)
train_pad_sequence = pad_sequences(training_sequence,maxlen = 200,truncating= 'post',padding = 'pre')
test_pad_sequence = pad_sequences(testing_sequence,maxlen = 200,truncating= 'post',padding = 'pre')
print('Total Unique Words : {}'.format(len(word_index)))


# # Using glove vectors for embedding

# In[ ]:


embedded_words = {}
with open('../input/glove-global-vectors-for-word-representation/glove.6B.200d.txt') as file:
    for line in file:
        words, coeff = line.split(maxsplit=1)
        coeff = np.array(coeff.split(),dtype = float)
        embedded_words[words] = coeff


# In[ ]:


embedding_matrix = np.zeros((len(word_index) + 1,200))
for word, i in word_index.items():
    embedding_vector = embedded_words.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# # Creating The Model
# layer1: Embedding Layer using glove weights 
# 
# layer2: Using a Bidirectional LSTM
# 
# layer3: A dropout Layer
# 
# layer4: A Dense layer of 256 neurons with 'relu' activation
# 
# layer5: A Dense Layer of 128 neurons with 'relu' activation
# 
# layer6: Again a dropout layer. 
# 
# layer7: Sigmoid activation layer to classify it positive and negative.
# 

# In[ ]:


model = tf.keras.Sequential([tf.keras.layers.Embedding(len(word_index) + 1,200,weights=[embedding_matrix],input_length=200,
                            trainable=False),
                             tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
                             tf.keras.layers.Dropout(0.8),
                             tf.keras.layers.Dense(256,activation = 'relu',),
                             tf.keras.layers.Dense(128,activation = 'relu'),
                             tf.keras.layers.Dropout(0.8),
                             tf.keras.layers.Dense(1,activation = tf.nn.sigmoid)])
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')


# In[ ]:


model.summary()


# In[ ]:


model.compile(loss = tf.keras.losses.BinaryCrossentropy() , optimizer='Adam' , metrics = 'accuracy')
history = model.fit(train_pad_sequence,training_labels,epochs = 30, validation_data=(test_pad_sequence,testing_labels),
                   callbacks=[mcp_save])


# In[ ]:


tf.keras.Model.save_weights(model, filepath='weight.hdf5')


# # Plotting Accuracy and Losses

# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)

plt.show()


# In[ ]:


print('Training Accuracy: {}'.format(max(acc)))
print('Validation Accuracy: {}'.format(max(val_acc)))


# # Conclusion
# 
# 1 - We have great accuracy and we can increase it training for much longer and tune other hyperparameters
# 2 - DNN LSTM have a deep impact on NLP problems and we can see that this model performs quite well.

# In[ ]:




