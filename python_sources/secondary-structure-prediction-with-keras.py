#!/usr/bin/env python
# coding: utf-8

# # Load Data
# To show proof of concept only considering sequences shorter than 32 amino acids. Also experiments suggest to apply a sliding overlapping window of size 3 to the input seqeunces. In the sense of  NLP this artificially creates a corpus consisting of much more different words (>7000 different ngrams compared to 20 amino acids). There has been a lot of research in respresenting protein sequences as sentences of  biological words. 

# In[1]:


import os
os.listdir('../input/')


# In[28]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

df1 = pd.read_csv('../input/protein-secondary-structure/2018-06-06-pdb-intersect-pisces.csv')
df1 = df1[~df1.has_nonstd_aa]
df2 = pd.read_csv('../input/protein-secondary-structure/2018-06-06-ss.cleaned.csv')
df2 = df2[~df2.has_nonstd_aa]

df2.len.hist(bins=100, alpha=.5)
df1.len.hist(bins=100)
plt.legend(['large', 'small'])
plt.ylim(0,2000)
print(df1.shape,df2.shape)


# In[15]:


df1.columns, df2.columns


# In[29]:


import numpy as np
from sklearn.model_selection import train_test_split

pdb_ids = np.unique(df1.pdb_id)
train_pdb_ids, test_pdb_ids = train_test_split(pdb_ids, test_size=.33)
pdb_ids.shape, train_pdb_ids.shape, test_pdb_ids.shape


# In[41]:


train1 = df1[df1.pdb_id.isin(train_pdb_ids)]
train2 = df2[df2.pdb_id.isin(train_pdb_ids)]
test1 = df1[df1.pdb_id.isin(test_pdb_ids)]
test2 = df2[df2.pdb_id.isin(test_pdb_ids)]
train1.shape, train2.shape, test1.shape, test2.shape


# # Preprocessing
# use preprocessing tools for text from keras to encode input sequence as word rank numbers  and target sequence as one hot. To ensure easy to use training and testing, all sequences are padded with zeros to the maximum sequence length (in our case 32).

# In[ ]:





# In[42]:


from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

maxlen_seq = 1024
X_train, X_test, y_train, y_test = train2.seq, test2.seq, train2.sst8, test2.sst8

tokenizer_encoder = Tokenizer(char_level=True)
tokenizer_encoder.fit_on_texts(input_seqs)
X_train = tokenizer_encoder.texts_to_sequences(X_train)
X_train = sequence.pad_sequences(X_train, maxlen=maxlen_seq, padding='post')
X_train = to_categorical(X_train, num_classes=len(tokenizer_encoder.word_index)+1)
X_test = tokenizer_encoder.texts_to_sequences(X_test)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen_seq, padding='post')
X_test = to_categorical(X_test, num_classes=len(tokenizer_encoder.word_index)+1)

tokenizer_decoder = Tokenizer(char_level=True)
tokenizer_decoder.fit_on_texts(target_seqs)

y_train = tokenizer_decoder.texts_to_sequences(y_train)
y_train = sequence.pad_sequences(y_train, maxlen=maxlen_seq, padding='post')
y_train = to_categorical(y_train, num_classes=len(tokenizer_decoder.word_index)+1)
y_test = tokenizer_decoder.texts_to_sequences(y_test)
y_test = sequence.pad_sequences(y_test, maxlen=maxlen_seq, padding='post')
y_test = to_categorical(y_test, num_classes=len(tokenizer_decoder.word_index)+1)

X_train.shape, y_train.shape, X_test.shape, y_test.shape


# # Build model
# This example is motivated by the NLP-task of POS-tagging. 

# In[52]:


from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional, Conv1D, MaxPool1D, UpSampling1D, Dropout, concatenate
from keras import optimizers

n_words = len(tokenizer_encoder.word_index) + 1
n_tags = len(tokenizer_decoder.word_index) + 1
print(n_words, n_tags)

inputs = Input(shape=(maxlen_seq,n_words))
conv1 = Conv1D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
conv1 = Conv1D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
pool1 = MaxPool1D(pool_size=(2))(conv1)
conv2 = Conv1D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
conv2 = Conv1D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
pool2 = MaxPool1D(pool_size=(2))(conv2)
conv3 = Conv1D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
conv3 = Conv1D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
pool3 = MaxPool1D(pool_size=(2))(conv3)
conv4 = Conv1D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
conv4 = Conv1D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
drop4 = Dropout(0.5)(conv4)
pool4 = MaxPool1D(pool_size=(2))(drop4)

conv5 = Conv1D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
conv5 = Conv1D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
drop5 = Dropout(0.5)(conv5)

up6 = Conv1D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling1D(size = (2))(drop5))
merge6 = concatenate([drop4,up6], axis = 2)
conv6 = Conv1D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
conv6 = Conv1D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

up7 = Conv1D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling1D(size = (2))(conv6))
merge7 = concatenate([conv3,up7], axis = 2)
conv7 = Conv1D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
conv7 = Conv1D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

up8 = Conv1D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling1D(size = (2))(conv7))
merge8 = concatenate([conv2,up8], axis = 2)
conv8 = Conv1D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
conv8 = Conv1D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

up9 = Conv1D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling1D(size = (2))(conv8))
merge9 = concatenate([conv1,up9], axis = 2)
conv9 = Conv1D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
conv9 = Conv1D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
conv9 = Conv1D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
conv10 = Conv1D(n_tags, 1, activation = 'softmax')(conv9)

model = Model(input = inputs, output = conv10)
model.summary()

model.compile(optimizer = optimizers.Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])


# # Train and evaluate model
# The model is trained such that the categorical crossentropy is minimized. For evalutation also Q3-accuracy is computed, by computing the accuracy only for coding characters. 

# In[53]:


from sklearn.model_selection import train_test_split
from keras.metrics import categorical_accuracy
from keras import backend  as K
import tensorflow as tf

def q3_acc(y_true, y_pred):
    y = tf.argmax(y_true, axis=-1)
    y_ = tf.argmax(y_pred, axis=-1)
    mask = tf.greater(y, 0)
    return K.cast(K.equal(tf.boolean_mask(y, mask), tf.boolean_mask(y_, mask)), K.floatx())

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", q3_acc])

model.fit(X_train, y_train, batch_size=64, epochs=15, validation_data=(X_test, y_test), verbose=1)


# # Results
# For this small example, the model trained very fast and achieves quite reasonable Q3-accuaracies. In the following three training and three testing examples are shown. Each image shows the softmax activation (red) on top of the underlying one hot ground truth (blue).

# In[ ]:


import biopython


# In[ ]:


def onehot_to_seq(oh_seq, index):
    s = ''
    for o in oh_seq:
        i = np.argmax(o)
        if i != 0:
            s += index[i]
        else:
            break
    return s

def plot_results(x, y, y_):
    print("---")
    print("Input: " + str(onehot_to_seq(x, revsere_encoder_index).upper()))
    print("Target: " + str(onehot_to_seq(y, revsere_decoder_index).upper()))
    print("Result: " + str(onehot_to_seq(y_, revsere_decoder_index).upper()))
    fig = plt.figure(figsize=(10,2))
    plt.imshow(y.T, cmap='Blues')
    plt.imshow(y_.T, cmap='Reds', alpha=.5)
    plt.yticks(range(4), [' '] + [revsere_decoder_index[i+1].upper() for i in range(3)])
    plt.show()
    
revsere_decoder_index = {value:key for key,value in tokenizer_decoder.word_index.items()}
revsere_encoder_index = {value:key for key,value in tokenizer_encoder.word_index.items()}

N=3
y_train_pred = model.predict(X_train[:N])
y_test_pred = model.predict(X_test[:N])
print('training')
for i in range(N):
    plot_results(X_train[i], y_train[i], y_train_pred[i])
print('testing')
for i in range(N):
    plot_results(X_test[i], y_test[i], y_test_pred[i])

