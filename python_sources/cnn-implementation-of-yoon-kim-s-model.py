#!/usr/bin/env python
# coding: utf-8

# ## Keras implementation of Yoon Kim's model for sentence classification

# ##### The following is a keras implementation of Yoon Kim's convolutional neural network model for sentence classification from the paper: https://arxiv.org/abs/1408.5882

# ##### The basic idea is to use a convolutional neural network where different convolutions are used to produce different n-gram-like filters to determine the sentiment of a given text. Using the "glove" pre-trained embeddings and manually setting different kernels sizes, the hope is to capture strong single phrases in sentences that results in a particular outcome.

# In[ ]:


import pandas as pd
import re
import numpy as np
from keras.preprocessing import sequence
from keras.regularizers import l2
from keras.models import Model
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Dense, GlobalMaxPooling1D, Activation, Dropout, GaussianNoise
from keras.layers import Embedding, Input, BatchNormalization, SpatialDropout1D, Conv1D
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from pandas_summary import DataFrameSummary 
from IPython.display import display
import itertools
from nltk.corpus import words
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# ##### Determine dimension of embedding vector, max size of vocabulary and max length of sentence (crop the rest)

# In[ ]:


# Set parameters
embed_size   = 50    # how big is each word vector
max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
maxlen       = 100   # max number of words in a comment to use 


# ##### Load data...

# In[1]:


# Load data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

list_sentences_train = train["comment_text"].fillna("_NaN_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("_NaN_").values


# ##### Tokenize sentences, convert to integers and pad sentences < 100 words

# In[ ]:


# Pad sentences and convert to integers
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

X_train = pad_sequences(list_tokenized_train, maxlen=maxlen, padding='post')
X_test = pad_sequences(list_tokenized_test, maxlen=maxlen, padding='post')


# ##### Load "glove" pre-trained embeddings and construct vocabulary dictionary

# In[ ]:


# Read the glove word vectors (space delimited strings) into a dictionary from word->vector
def get_coefs(word,*arr): 
    return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open('./data/glove.6B.50d.txt'))


# ##### Create embedding matrix and initialize space for new words not present in "glove"

# In[ ]:


# Create embeddings matrix
all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()

# Create embedding matrix using our vocabulary
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))

# Initialize embedding matrix
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

# Loop through each word and get its embedding vector
for word, i in word_index.items():
    if i >= max_features: 
        continue # Skip words appearing less than the minimum allowed
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: 
        embedding_matrix[i] = embedding_vector


# ##### Set no. of convolution filters and weigh the outcome variable in order to balance.
# - 128 filters are used for each convolution. I.e. with a kernel size of 3, 128 tri grams are constructed each representing a specific feature. With a kernel size of 4, 128 4-grams are constructed and so on..

# In[ ]:


# Initialize parameters
conv_filters = 128 # No. filters to use for each convolution
weight_vec = list(np.max(np.sum(y, axis=0))/np.sum(y, axis=0))
class_weight = {i: weight_vec[i] for i in range(6)}


# ##### Construct Convolutional Neural Network

# In[ ]:


inp = Input(shape=(X_train.shape[1],), dtype='int64')
emb = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)

# Specify each convolution layer and their kernel siz i.e. n-grams 
conv1_1 = Conv1D(filters=conv_filters, kernel_size=3)(emb)
btch1_1 = BatchNormalization()(conv1_1)
drp1_1  = Dropout(0.2)(btch1_1)
actv1_1 = Activation('relu')(drp1_1)
glmp1_1 = GlobalMaxPooling1D()(actv1_1)

conv1_2 = Conv1D(filters=conv_filters, kernel_size=4)(emb)
btch1_2 = BatchNormalization()(conv1_2)
drp1_2  = Dropout(0.2)(btch1_2)
actv1_2 = Activation('relu')(drp1_2)
glmp1_2 = GlobalMaxPooling1D()(actv1_2)

conv1_3 = Conv1D(filters=conv_filters, kernel_size=5)(emb)
btch1_3 = BatchNormalization()(conv1_3)
drp1_3  = Dropout(0.2)(btch1_3)
actv1_3 = Activation('relu')(drp1_3)
glmp1_3 = GlobalMaxPooling1D()(actv1_3)

conv1_4 = Conv1D(filters=conv_filters, kernel_size=6)(emb)
btch1_4 = BatchNormalization()(conv1_4)
drp1_4  = Dropout(0.2)(btch1_4)
actv1_4 = Activation('relu')(drp1_4)
glmp1_4 = GlobalMaxPooling1D()(actv1_4)

# Gather all convolution layers
cnct = concatenate([glmp1_1, glmp1_2, glmp1_3, glmp1_4], axis=1)
drp1 = Dropout(0.2)(cnct)

dns1  = Dense(32, activation='relu')(drp1)
btch1 = BatchNormalization()(dns1)
drp2  = Dropout(0.2)(btch1)

out = Dense(y.shape[1], activation='sigmoid')(drp2)


# In[ ]:


# Compile
model = Model(inputs=inp, outputs=out)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


# Estimate model
model.fit(X_train, y, validation_split=0.1, epochs=2, batch_size=32, shuffle=True, class_weight=class_weight)


# ##### Predict and finally submit

# In[ ]:


# Predict
preds = model.predict(X_test)


# In[ ]:


# Create submission
submid = pd.DataFrame({'id': test["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = list_classes)], axis=1)
submission.to_csv('conv_glove_simple_sub.csv', index=False)

