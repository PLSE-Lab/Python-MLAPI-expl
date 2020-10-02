#!/usr/bin/env python
# coding: utf-8

# In this kernel, we will build different models which can predict the number of stars given by each review. We will then compare the accuracies of all models to find out the best one. Let's get started ! :) 

# In[ ]:


#Load the csv file into a dataframe
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('../input/100k-courseras-course-reviews-dataset/reviews.csv')
df.head() # What's in there?


# In[ ]:


# We don't need the Id column. Let's drop it !
df.drop(columns=['Id'],inplace=True)
df.head()


# In[ ]:


# How many data points?
df.shape #(rows, columns)


# In[ ]:


df.isna().any() # Any null values?


# In[ ]:


df.dropna(how='any',inplace=True) # drop any null values!


# **Text pre-processing**
# 
# We are going to use Tokenizer API provided by keras for text pre-processing. Our models cannot accept the texts as input. So, we need to convert them into integers first. There could be a large number of words but we are going to deal with only the first 8000 words ;) 

# In[ ]:


from keras.preprocessing.text import Tokenizer
MAX_WORDS = 8000 
t = Tokenizer(num_words=MAX_WORDS)
t.fit_on_texts(df['Review']) # assigns unique int to all the words
word_index=t.word_index
word_index # Can you see that?! It is a dictionary of words to numbers


# In[ ]:


df['Review'][0] # The first review


# In[ ]:


df['Review']=t.texts_to_sequences(df['Review']) #Let's apply the transformation on our reviews !


# In[ ]:


df['Review'][0] # Did you notice something? [good and interesting] is now [16, 2, 39]


# In[ ]:


# Split our data into training and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df['Review'],df['Label'], test_size = 0.2)


# In[ ]:


# What is the max and min length of reviews?
review_length = [len(x) for x in X_train]
print(max(review_length))
print(min(review_length))


# In[ ]:


# We need to provide a fixed length input so the length of all the reviews should be the same. 
# We are going to take the first 500 words of each review to predict the number of stars.
from keras.preprocessing import sequence
input_limit = 500 # first 500 words only of each review to be considered
X_train = sequence.pad_sequences(X_train,maxlen=input_limit) # pad with 0 if length is less than 500
X_test = sequence.pad_sequences(X_test,maxlen=input_limit) # pad with 0 if length is less than 500


# In[ ]:


y_train.unique() # how many stars?


# In[ ]:


# Ofcourse we need to do one hot encoding here! 
print(y_train[0]) #before one hot encoding
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train-1,num_classes=5)
y_test = np_utils.to_categorical(y_test-1,num_classes=5)
y_train[0] # How label looks after one encoding


# In[ ]:


# Split the test set into test and validation set.
X_test,X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5)


# **MLP**
# 
# Let's begin with using a MLP to build our model and see how it performs! Btw, keras provides an **embedding layer** which we are going to use here.
# Embedding layers are used to represent each word as a vector in a pre-defined vector space(we have define the number of elements to be in this vector). **Word embeddings** are used to learn the semantic relationship among words. For example, in a n-dimensional vector space, the words *apple* and *orange* will be closer to each other than the word *car*. It tries to find the contextual meaning and helps to learn better about the given text.

# In[ ]:


from keras.models import Sequential
from keras.layers import Embedding, Dense, Flatten

model = Sequential()
# We choose to represnt each word as a 16 element vector=output_dim
# Note that the output of Embedding layer will be MAX_WORDS*16 i.e. each of the MAX_WORDS will be
# represented as a 16 element vector.  
model.add(Embedding(input_dim=MAX_WORDS, output_dim=16,input_length=input_limit))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(5,activation='softmax'))
model.compile(optimizer='adam',metrics=['accuracy'],loss='categorical_crossentropy')
model.summary()


# In[ ]:


#Let's fit the model
from keras.callbacks import ModelCheckpoint
model_checkpoint = ModelCheckpoint('best.hdf5',save_best_only=True)
model.fit(X_train,y_train, validation_data=(X_valid,y_valid),batch_size=64, epochs=10, callbacks=[model_checkpoint])


# In[ ]:


# Time to check for accuracy
model.load_weights('best.hdf5')
score= model.evaluate(X_test,y_test)
score[1]


# **CNN**
# 
# Here, we are going to keep things nearly the same  as above except for adding some convolutional layers.

# In[ ]:


from keras.layers import Conv1D,MaxPooling1D

model = Sequential()
model.add(Embedding(input_dim=MAX_WORDS,output_dim=256,input_length=500))
model.add(Conv1D(filters=256,kernel_size=2,padding='same',activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128,kernel_size=2,padding='same',activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64,kernel_size=2,padding='same',activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=32,kernel_size=2,padding='same',activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(5,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


# In[ ]:


from keras.callbacks import ModelCheckpoint
model_checkpoint = ModelCheckpoint('best.hdf5',save_best_only=True)
model.fit(X_train,y_train, validation_data=(X_valid,y_valid),batch_size=64, epochs=10, callbacks=[model_checkpoint])


# In[ ]:


model.load_weights('best.hdf5')
score= model.evaluate(X_test,y_test)
score[1]


# **Pre-trained embedding - Word2Vec **
# 
# In both the previous models, along with learning the params for other layers, word embedding values are also learnt i.e. the word vector values get updated with each epoch. Now, we are going to use some embeddings which have already learned for some thousand words like Glove. Note that we are not going to update the values of the already learned word embeddings so we used the parameter `trainable=False` while adding the embedding layer. Major steps to follow are :
# 
# 1. Load the word embedding.
# 2. Create an embedding matrix such that element i represents the word vector for the word whose word_index is i.
# 3. Load this embedding matrix into the embedding layer with weights frozen.
# 4. Build a CNN on top of it.

# In[ ]:


import os
embedding_index={}
f = open(os.path.join('../input/glove-100-dimension','glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float')
    embedding_index[word] = coefs
print('Found %s word vectors' % len(embedding_index))


# In[ ]:


EMBED_DIM=100
embedding_matrix = np.zeros((MAX_WORDS, EMBED_DIM))
print(embedding_matrix.shape)
for word, index in word_index.items():
    if index > MAX_WORDS-1:
        continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector


# In[ ]:


embedding_matrix.shape


# In[ ]:


from keras.layers import Embedding
embedding_layer = Embedding(MAX_WORDS,
                            EMBED_DIM,
                            weights=[embedding_matrix],
                            input_length=input_limit,
                            trainable=False)


# In[ ]:


from keras.layers import Conv1D, MaxPooling1D,Flatten,Dense,GlobalMaxPooling1D, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(filters=128,kernel_size=5,padding='same',activation='relu'))
model.add(MaxPooling1D(pool_size=5))
model.add(Conv1D(filters=128,kernel_size=5,padding='same',activation='relu'))
model.add(MaxPooling1D(pool_size=5))
model.add(Conv1D(filters=128,kernel_size=5,padding='same',activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(128,activation='relu'))
model.add(Dense(5,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.01, momentum=0.9),metrics=['acc'])
model.summary()


# In[ ]:


from keras.callbacks import ModelCheckpoint
model_checkpoint = ModelCheckpoint('best.hdf5',save_best_only=True)
model.fit(X_train,y_train, validation_data=(X_valid,y_valid),batch_size=128, epochs=10, callbacks=[model_checkpoint])


# In[ ]:


model.load_weights('best.hdf5')
score= model.evaluate(X_test,y_test)
score[1]


# To conclude, basic MLP and CNN perform almost the same ~80% as using Glove embedding :) 
# 
# Please upvote if you found this useful!
# 
