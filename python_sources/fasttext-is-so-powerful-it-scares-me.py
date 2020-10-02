#!/usr/bin/env python
# coding: utf-8

# # FastText is so Powerful It Scares Me

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
from time import time
import json

import re
import string
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from gensim.models import FastText

from keras.layers import Input, Dense, Embedding, Flatten, Dropout, SpatialDropout1D # General
from keras.layers import CuDNNLSTM, Bidirectional # LSTM-RNN
from keras.optimizers import Adam

from keras import backend as K
from keras.callbacks import EarlyStopping

import tensorflow as tf

# Evaluation
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/ndsc-beginner/train.csv')


# ### Remove Numbers and Punctuations

# In[ ]:


table = str.maketrans('','', string.punctuation)

def removeNumbersAndPunctuations(text):
    text = text.translate(table)
    text = re.sub(r'\d+', '', text)
    return text


# In[ ]:


df['title'] = df['title'].apply(removeNumbersAndPunctuations)


# ### Split dataset into Train and Test

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df['title'], df['Category'], test_size=0.16, random_state=42)


# ### Load Embeddings
# Our embeddings are trained on the training dataset using FastText's Skipgram model. For more details on FastText, you can visit the link here: https://fasttext.cc/

# In[ ]:


print('loading word embeddings...')
embeddings_index = {}
f = open('../input/ftembeddings300all/ftembeddings300all.txt', encoding='utf-8')
for line in tqdm(f):
    values = line.rstrip().rsplit(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('found %s word vectors' % len(embeddings_index))


# ### Create One-Hot for Train and Test Ys

# In[ ]:


y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)


# ### Constants

# In[ ]:


NUM_CATEGORIES = 58
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 20000
EMBED_DIM = 300
HIDDEN = 256


# ### Create Sequence Matrices (Features) for Train, Test

# In[ ]:


tok = Tokenizer(num_words=MAX_NB_WORDS, lower=True) 
tok.fit_on_texts(X_train)


# In[ ]:


word_index = tok.word_index
print('Found %s unique tokens.' % len(word_index))


# In[ ]:


sequences = tok.texts_to_sequences(X_train)
train_dtm = sequence.pad_sequences(sequences,maxlen=MAX_SEQUENCE_LENGTH)

test_sequences = tok.texts_to_sequences(X_test)
test_dtm = sequence.pad_sequences(test_sequences,maxlen=MAX_SEQUENCE_LENGTH)


# In[ ]:


print('Shape of Train DTM:', train_dtm.shape)


# ### Create Embedding Matrix

# In[ ]:


print('preparing embedding matrix...')
words_not_found = []
NUM_WORDS = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((NUM_WORDS, EMBED_DIM))
for word, i in word_index.items():
    if i >= NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if (embedding_vector is not None) and len(embedding_vector) > 0:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
        words_not_found.append(word)
print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
print("sample words not found: ", np.random.choice(words_not_found, 10))


# ### Create RNN Model

# In[ ]:


def RNN_Model():
    text_sequence = Input(shape=(MAX_SEQUENCE_LENGTH,), name='TEXT_SEQUENCE_INPUT')
    
    rnn_layer = Embedding(NUM_WORDS, EMBED_DIM, weights=[embedding_matrix], trainable=False, name='EMBEDDING')(text_sequence) 
    rnn_layer = SpatialDropout1D(0.5, name='EMBEDDING_DROPOUT')(rnn_layer)
    rnn_layer = Bidirectional(CuDNNLSTM(HIDDEN, return_sequences=True), name='BILSTM_LAYER1')(rnn_layer)
    rnn_layer = Bidirectional(CuDNNLSTM(HIDDEN), name='BILSTM_LAYER2')(rnn_layer)
    rnn_layer = Dropout(0.5,name='RNN_DROPOUT')(rnn_layer)

    output = Dense(NUM_CATEGORIES, activation='softmax', name='OUTPUT')(rnn_layer)
    model = Model(inputs=text_sequence, outputs=output)
    
    return model


# In[ ]:


K.clear_session()
model = RNN_Model()
model.summary()


# ## Train with Frozen Embedding Layer
# Currently, the Embedding Layer's trainable parameter is set to False. The weights of the neural network, less the embedding matrix, is random at initialization. What we want to do here is transfer the learned context and semantic meaning within the pre-trained embeddings to this neural network. 
# If the trainable parameter is set to True when the neural network is still untrained, it would confuse the pre-trained embedding matrix, with the possibility of causing it to lose most of things it had learnt. 

# In[ ]:


ea = EarlyStopping(monitor='val_categorical_accuracy', patience=3, restore_best_weights=True)
adam = Adam(lr=0.001, decay=0.000049, epsilon=1e-8)


# In[ ]:


model.compile(loss='categorical_crossentropy',optimizer=adam, metrics=['categorical_accuracy'])
model.fit(train_dtm, y_train, batch_size=128, epochs=30, validation_data=(test_dtm,y_test), verbose=1, callbacks=[ea])


# ## Continue Training with Unfrozen Embedding Layer
# Now that the neural network is more or less trained, we unfreeze the embedding layer in hopes that it could continue to train further.

# In[ ]:


ea2 = EarlyStopping(monitor='val_categorical_accuracy', patience=3, restore_best_weights=True)
adam2 = Adam(lr=0.001, decay=0.00006, epsilon=1e-8)


# In[ ]:


model.layers[1].trainable = True
model.compile(loss='categorical_crossentropy',optimizer=adam2, metrics=['categorical_accuracy'])
model.fit(train_dtm, y_train, batch_size=128, epochs=20, validation_data=(test_dtm,y_test), verbose=1, callbacks=[ea2])


# In[ ]:


model.evaluate(test_dtm, y_test)


# In[ ]:


y_pred = [np.argmax(pred) for pred in model.predict(test_dtm)]
y_truth = [np.argmax(truth) for truth in y_test.values]


# In[ ]:


with open('../input/ndsc-beginner/categories.json', 'rb') as handle:
    catNames = json.load(handle)

catNameMapper = {}
for category in catNames.keys():
    for key, value in catNames[category].items():
        catNameMapper[value] = key


# In[ ]:


catNameLabelsSorted = ['SPC', 'Icherry', 'Alcatel', 'Maxtron', 'Strawberry', 'Honor', 'Infinix', 'Realme', 
                       'Sharp', 'Smartfren', 'Motorola', 'Mito', 'Brandcode', 'Evercoss', 'Huawei', 
                       'Blackberry', 'Advan', 'Lenovo', 'Nokia', 'Sony', 'Asus', 'Vivo', 'Xiaomi', 'Oppo', 
                       'Iphone', 'Samsung', 'Others Mobile & Tablet', 'Big Size Top', 'Wedding Dress', 
                       'Others', 'Crop Top ', 'Big Size Dress', 'Tanktop', 'A Line Dress', 'Party Dress', 
                       'Bodycon Dress', 'Shirt', 'Maxi Dress', 'Blouse\xa0', 'Tshirt', 'Casual Dress', 
                       'Lip Liner', 'Setting Spray', 'Contour', 'Other Lip Cosmetics', 'Lip Gloss', 'Lip Tint', 
                       'Face Palette', 'Bronzer', 'Highlighter', 'Primer', 'Blush On', 'Concealer', 'Lipstick', 
                       'Foundation', 'Other Face Cosmetics', 'BB & CC Cream', 'Powder']


# In[ ]:


catNamePred = list(map(lambda x: catNameMapper[x], y_pred))
catNameActual = list(map(lambda x: catNameMapper[x], y_truth))


# In[ ]:


confMat = confusion_matrix(catNamePred, catNameActual, labels=catNameLabelsSorted)


# In[ ]:


fig, ax = plt.subplots(figsize=(30,30))
sns.heatmap(confMat, annot=True, fmt='d', xticklabels=catNameLabelsSorted, yticklabels=catNameLabelsSorted)
plt.ylabel('PREDICTED')
plt.xlabel('ACTUAL')
plt.show()


# We see that the model finds it hard to distinguish the dresses, and the Other Mobile and Tablets items are spread out across the rest of the categories. 

# ## Submission

# In[ ]:


test_data = pd.read_csv('../input/ndsc-beginner/test.csv')
test_data['title'] = test_data['title'].apply(removeNumbersAndPunctuations)

test_sequences = tok.texts_to_sequences(test_data.title)
test_dtm = sequence.pad_sequences(test_sequences,maxlen=MAX_SEQUENCE_LENGTH)

y_pred = [np.argmax(pred) for pred in model.predict(test_dtm)]
test_data['Category'] = y_pred


# In[ ]:


test_data


# In[ ]:


df_submit = test_data[['itemid', 'Category']].copy()
df_submit.to_csv('submission_svc.csv', index=False)

