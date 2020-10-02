#!/usr/bin/env python
# coding: utf-8

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
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from gensim.models import FastText

from keras import backend as K
from keras.callbacks import EarlyStopping

import tensorflow as tf

from keras.layers import Input, Dense, Embedding, Flatten, Dropout, SpatialDropout1D # General
from keras.layers import CuDNNLSTM, Bidirectional # LSTM-RNN
from keras.optimizers import Adam

# Evaluation
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/ndsc-beginner/train.csv')


# In[ ]:


df.head()


# ## Remove Numbers and Punctuations

# In[ ]:


table = str.maketrans('','', string.punctuation)

def removeNumbersAndPunctuations(text):
    text = text.translate(table)
    text = re.sub(r'\d+', '', text)
    return text


# In[ ]:


df['title'] = df['title'].apply(removeNumbersAndPunctuations)


# In[ ]:


X, y = {}, {}
tokenizers = {}
MAX_NB_WORDS = 20000

for typ in ['mobile', 'fashion', 'beauty']:    
    sdf = df[df['image_path'].str.contains(typ)]
    
    X[typ], y[typ] = {}, {}
    X[typ]['train'], X[typ]['test'], y[typ]['train'], y[typ]['test'] = train_test_split(sdf['title'], sdf['Category'], test_size=0.16, random_state=42)

    tok = Tokenizer(num_words=MAX_NB_WORDS, lower=True) 
    tok.fit_on_texts(X[typ]['train'])
    tokenizers[typ] = tok
    

valaccs = {}
valtruth = {}
valpreds = {}


# In[ ]:


def getModel(X, y, typ, tokenizer):
    print(f'Processing for {typ} dataset...')
    
    # Split dataset into Train and Test    
    X_train, X_test, y_train, y_test = X['train'], X['test'], y['train'], y['test']
    
    # Load embeddings
    print(f'Loading word embeddings for {typ} dataset...')
    embeddings_index = {}
    f = open(f'../input/creating-fasttext-embeddings/ftembeddings300{typ}.txt', encoding='utf-8')
    for line in tqdm(f):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print(f'found {len(embeddings_index)} word vectors for {typ} dataset')
    
    # One-hot y datasets
    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)
    
    # Constants
    NUM_CATEGORIES = y['train'].nunique()
    MAX_SEQUENCE_LENGTH = 30
    MAX_NB_WORDS = 20000
    EMBED_DIM = 300
    HIDDEN = 256
    
    print(f'Creating sequence matrices for {typ}')
    # Create Sequence Matrices
    tok = tokenizer
    word_index = tok.word_index
    
    sequences = tok.texts_to_sequences(X_train)
    train_dtm = sequence.pad_sequences(sequences,maxlen=MAX_SEQUENCE_LENGTH)

    test_sequences = tok.texts_to_sequences(X_test)
    test_dtm = sequence.pad_sequences(test_sequences,maxlen=MAX_SEQUENCE_LENGTH)
    
    # Prepare embedding matrix
    print(f'Preparing embedding matrix for {typ} dataset...')
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
    
    WEIGHTS = embedding_matrix
    model = RNN_Model(NUM_CATEGORIES, NUM_WORDS, MAX_SEQUENCE_LENGTH, EMBED_DIM, HIDDEN, WEIGHTS)

    print(f'Training for {typ} dataset with frozen embedding layer...')
    
    ea = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    adam = Adam(lr=0.001, decay=0.000049, epsilon=1e-8)
    
    model.compile(loss='categorical_crossentropy',optimizer=adam, metrics=['categorical_accuracy'])
    model.fit(train_dtm, y_train, batch_size=128, epochs=20, validation_data=(test_dtm,y_test), verbose=1, callbacks=[ea])    
    
    print(f'Training for {typ} dataset with unfrozen embedding layer...')
    
    ea2 = EarlyStopping(monitor='val_categorical_accuracy', patience=3, restore_best_weights=True)
    adam2 = Adam(lr=0.001, decay=0.00006, epsilon=1e-8)
    
    model.layers[1].trainable = True
    model.compile(loss='categorical_crossentropy',optimizer=adam2, metrics=['categorical_accuracy'])
    model.fit(train_dtm, y_train, batch_size=128, epochs=20, validation_data=(test_dtm,y_test), verbose=1, callbacks=[ea2])
    
    valaccs[typ] = model.evaluate(test_dtm, y_test)
    valtruth[typ] = [np.argmax(truth) + y_test.columns[0] for truth in y_test.values]
    valpreds[typ] = [np.argmax(pred)+ y_test.columns[0] for pred in model.predict(test_dtm)]
    return model


# ### Create RNN Model

# In[ ]:


def RNN_Model(NUM_CATEGORIES, NUM_WORDS, MAX_SEQUENCE_LENGTH, EMBED_DIM, HIDDEN, WEIGHTS):
    text_sequence = Input(shape=(MAX_SEQUENCE_LENGTH,), name='TEXT_SEQUENCE_INPUT')
    
    rnn_layer = Embedding(NUM_WORDS, EMBED_DIM, weights=[WEIGHTS], trainable=False, name='EMBEDDING')(text_sequence)
    rnn_layer = SpatialDropout1D(0.5, name='EMBEDDING_DROPOUT')(rnn_layer)
    rnn_layer = Bidirectional(CuDNNLSTM(HIDDEN, return_sequences=True), name='BILSTM_LAYER1')(rnn_layer)
    rnn_layer = Bidirectional(CuDNNLSTM(HIDDEN, return_sequences=True), name='BILSTM_LAYER2')(rnn_layer)
    rnn_layer = Flatten()(rnn_layer)
    rnn_layer = Dropout(0.4,name='RNN_DROPOUT')(rnn_layer)

    output = Dense(NUM_CATEGORIES, activation='softmax', name='OUTPUT')(rnn_layer)
    model = Model(inputs=text_sequence, outputs=output)
    
    return model


# In[ ]:


models = {}
for typ in ['mobile', 'fashion', 'beauty']:
    models[typ] = getModel(X[typ], y[typ], typ, tokenizers[typ])


# In[ ]:


y_truth, y_pred = [], []
for key in ['mobile', 'fashion', 'beauty']:
    y_truth.extend(valtruth[key])
    y_pred.extend(valpreds[key])


# In[ ]:


count = 0
for t,p in zip(y_truth, y_pred):
    if t == p:
        count += 1
print('Accuracy:', count/len(y_pred))


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


# ## Submission

# In[ ]:


test_data = pd.read_csv('../input/ndsc-beginner/test.csv')
test_data['title'] = test_data['title'].apply(removeNumbersAndPunctuations)
test_data['typ'] = test_data['image_path'].apply(lambda x: x[:x.index('_')])


# In[ ]:


s = {'beauty': 0, 'fashion': 17, 'mobile': 31}
preds = []
for ind, row in test_data.iterrows():
    tok = tokenizers[row['typ']]
    sequences = tok.texts_to_sequences([row['title']])
    test_dtm = sequence.pad_sequences(sequences,maxlen=30)
    preds.append(np.argmax(models[row['typ']].predict(test_dtm)) + s[row['typ']])


# In[ ]:


test_data['Category'] = preds


# In[ ]:


df_submit = test_data[['itemid', 'Category']].copy()
df_submit.to_csv('submission_svc.csv', index=False)

