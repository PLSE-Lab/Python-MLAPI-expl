#!/usr/bin/env python
# coding: utf-8

# # Project 3
# 
# 
# # Conversations Toxicity Detection
# 
# ## Preproccesing and CuDNNLSTM Model
# 
# Jigsaw Unintended Bias in Toxicity Classification 
# 
# Detect toxicity across a diverse range of conversations
# 
# 
# https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data#
# 

# # Import Libraries

# In[34]:


import gc
import re
import operator 

import numpy as np
import pandas as pd

from gensim.models import KeyedVectors

from sklearn import model_selection

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input, Dense, CuDNNGRU,concatenate, Bidirectional, SpatialDropout1D, Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import LeakyReLU, CuDNNLSTM
from keras.optimizers import RMSprop, Adam
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping

import seaborn as sns


# In[17]:


import os
print(os.listdir("../input"))


# # Import Data
# 

# In[18]:


train = pd.read_csv("../input/outprocess3/train_preprocess2.csv")
test = pd.read_csv("../input/outprocess3/test_preprocess2.csv")
print("Train shape : ",train.shape)
test.head()


# # Further Preparation

# In[19]:


train_orig = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv")
train_orig.head()


# In[20]:


train = pd.concat([train,train_orig[['target']]],axis=1)
train.head()


# In[21]:


del(train_orig)
gc.collect()


# Convert target to binary flag

# In[22]:


train['target'] = np.where(train['target'] >= 0.5, True, False)


# Split into train/validation sets

# In[23]:


train_df, validate_df = model_selection.train_test_split(train, test_size=0.1)
print('%d train comments, %d validate comments' % (len(train_df), len(validate_df)))


# In[24]:


train_df.head()


# Tokenize the text

# In[31]:


train_df['comment_text'].describe()


# In[36]:


MAX_NUM_WORDS = 100000
TOXICITY_COLUMN = 'target'
TEXT_COLUMN = 'comment_text'

# Create a text tokenizer.
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(train_df[['comment_text']])

# All comments must be truncated or padded to be the same length.
MAX_SEQUENCE_LENGTH = 256
def pad_text(texts, tokenizer):
    return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_SEQUENCE_LENGTH)


# Create our embedding matrix

# In[37]:


gc.collect()


# In[39]:


EMBEDDINGS_DIMENSION = 300
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1,EMBEDDINGS_DIMENSION))


# In[38]:


ft_common_crawl = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
embeddings_index = KeyedVectors.load_word2vec_format(ft_common_crawl)


# In[40]:


num_words_in_embedding = 0

for word, i in tokenizer.word_index.items():
    if word in embeddings_index.vocab:
        embedding_vector = embeddings_index[word]
        embedding_matrix[i] = embedding_vector        
        num_words_in_embedding += 1


# In[42]:


train_text = pad_text(train_df[[TEXT_COLUMN]], tokenizer)
train_labels = train_df[[TOXICITY_COLUMN]]
validate_text = pad_text(validate_df[[TEXT_COLUMN]], tokenizer)
validate_labels = validate_df[[TOXICITY_COLUMN]]


# In[43]:


gc.collect()


# # Model Architecture
# 
# Adding dropout / 1d conv / concatenated poolings based on the architecture presented @ https://www.kaggle.com/tunguz/bi-gru-cnn-poolings-gpu-kernel-version
# 
# Now based on: https://www.kaggle.com/samarthsarin/toxication-with-embeddings-and-keras-lstm

# In[ ]:


NODES = 64
vocab_size = len(tokenizer.word_index) + 1


model = Sequential()

model.add(Embedding(vocab_size,EMBEDDINGS_DIMENSION,input_length = MAX_SEQUENCE_LENGTH,weights = [embedding_matrix],trainable = False))

model.add(Bidirectional(CuDNNLSTM(100,return_sequences=True)))
model.add(Conv1D(64,7,padding='same'))
model.add(GlobalAveragePooling1D())

model.add(Dense(128))
model.add(LeakyReLU())

model.add(Dense(NODES,activation = 'relu'))

model.add(Dense(1,activation = 'sigmoid'))

model.summary()


# In[ ]:


model.compile(optimizer = 'adam',loss='binary_crossentropy',metrics = ['accuracy'])


# # Model Training

# In[ ]:


BATCH_SIZE = 1024
NUM_EPOCHS = 100


# In[ ]:


model.fit(
    train_text,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    validation_data=(validate_text, validate_labels),
    callbacks = [EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)])


# In[ ]:





# # Predict & Submit

# Let's submit this as our first submission. Once we have a reasonable pipeline setup, we can move on to looking at the competition metric in more detail.

# In[ ]:


submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv', index_col='id')
submission['prediction'] = model.predict(pad_text(test[TEXT_COLUMN], tokenizer))
submission.reset_index(drop=False, inplace=True)
submission.head()


# In[ ]:



submission.to_csv('submission.csv', index=False)

