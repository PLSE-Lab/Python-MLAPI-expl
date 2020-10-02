#!/usr/bin/env python
# coding: utf-8

# ## I.Preprocess

# In[32]:


import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import SpatialDropout1D, add, concatenate
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers


# In[33]:


Train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sub = pd.read_csv('../input/sample_submission.csv')


# In[34]:


train = Train


# In[35]:


IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]
AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
TEXT_COLUMN = 'comment_text'
TARGET_COLUMN = 'target'


# In[36]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
def preprocess(sentences):
    list_sentences = list(sentences)
    text_tokenized = [[word.lower() for word in word_tokenize(sen)] for sen in list_sentences]
    text_filtered = [[word for word in sen if not word in english_punctuations] for sen in text_tokenized]
    st = LancasterStemmer()
    texts_stemmed = [[st.stem(word) for word in sen] for sen in text_filtered]
    preprocessed_sentences = [" ".join(sen) for sen in texts_stemmed]
    return preprocessed_sentences


# In[46]:


y_train = train[TARGET_COLUMN].values
y_aux_train = train[AUX_COLUMNS].values
list_sentences_train = preprocess(train["comment_text"])
list_sentences_test = preprocess(test["comment_text"])


# In[47]:


max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)


# In[48]:


maxlen = 200
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)


# In[50]:


y = train[IDENTITY_COLUMNS+AUX_COLUMNS].fillna(0).values


# ## II. Build Model
# After finish prepocessing of data, we start to build our LSTM model

# In[51]:


inp = Input(shape=(maxlen, ))
embed_size = 128
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS


# In[52]:


x = Embedding(max_features, embed_size)(inp)
x = SpatialDropout1D(0.2)(x)
x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)


# In[53]:


hidden = concatenate([
        GlobalMaxPooling1D()(x),
        GlobalAveragePooling1D()(x),
    ])
hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
result = Dense(1, activation='sigmoid')(hidden)
aux_result = Dense(len(AUX_COLUMNS), activation='sigmoid')(hidden)


# In[54]:


model = Model(inputs=inp, outputs=[result, aux_result])
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


# ## III. Runtime

# In[55]:


batch_size = 32
epochs = 2
model.fit(X_t,[y_train, y_aux_train], batch_size=batch_size, epochs=epochs, validation_split=0.1)


# In[56]:


predictions = model.predict(X_te)


# In[57]:


probabilities = predictions[0]
output_df = pd.DataFrame(probabilities, columns=['prediction'])
merged_df =  pd.concat([test, output_df], axis=1)
submission = merged_df.drop(['comment_text'], axis=1)


# In[58]:


submission.to_csv("submission.csv", index=False)


# In[ ]:




