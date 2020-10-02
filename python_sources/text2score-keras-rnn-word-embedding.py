#!/usr/bin/env python
# coding: utf-8

# ## Intro

# In this script I use word embeddings and RNN to predict the deal_probability directly.  <br>
# The embeddings are learng as part of the training process. <br>
# Usiging FastText's pre train vectores in Russian, better results can be achieved<br>
# Pre-traiened FastText: https://s3-us-west-1.amazonaws.com/fasttext-vectors/word-vectors-v2/cc.ru.300.vec.gz

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from keras.preprocessing import sequence, text
from keras.models import Sequential
import keras.layers as layer 
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras import backend as K

from sklearn.model_selection import train_test_split


# In[2]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_all = pd.concat([df_train, df_test])
del(df_train, df_test)


# ## Setting up the text feature

# In[3]:


text_features = [ 'title', 'category_name', 'parent_category_name', 'description', 'param_1', 'param_2', 'param_3',]


# In[4]:


df_all = df_all[text_features + ['deal_probability']]


# In[5]:


df_all['text'] = ""
for text_col in text_features:
    df_all['text'] += " " + df_all[text_col].fillna("")
    
pattern = re.compile('[^(?u)\w\s]+')
df_all['text'] =df_all['text'].apply(lambda x: re.sub(pattern, "", x).lower())


# ## Tokenizing

# In[6]:


max_len = 30
tk = text.Tokenizer(num_words=50000)
tk.fit_on_texts(df_all['text'].str.lower().tolist())
X = tk.texts_to_sequences(df_all['text'].str.lower().values)
X = sequence.pad_sequences(X, maxlen=max_len)


# In[7]:


df_all.drop(text_features, axis=1, inplace=True)


# In[8]:


word_index = tk.word_index


# ## Train test split

# In[27]:


df_train = df_all[df_all['deal_probability'].notnull()]
X_train, X_val, y_train, y_val  = train_test_split(X[:len(df_train)], df_train['deal_probability'].values, test_size=0.01)


# ## Build keras model

# In[10]:


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


# In[48]:


def get_model():

    model = Sequential()
    model.add(layer.Embedding(len(word_index) + 1, 30, input_shape=(max_len,)))
    model.add(layer.LSTM(30, recurrent_dropout=0.2, dropout=0.2, kernel_regularizer=regularizers.l2(2e-5),
                activity_regularizer=regularizers.l1(2e-5)))
    
    model.add(layer.Dense(32,  kernel_regularizer=regularizers.l2(2e-5),
                activity_regularizer=regularizers.l1(2e-5)))
    model.add(layer.PReLU())
    model.add(layer.Dropout(0.2))
    model.add(layer.BatchNormalization())
    
    model.add(layer.Dense(32, kernel_regularizer=regularizers.l2(2e-5),
                activity_regularizer=regularizers.l1(2e-5)))
    model.add(layer.PReLU())
    model.add(layer.Dropout(0.2))
    model.add(layer.BatchNormalization())

    
    model.add(layer.Dense(1))
    model.add(layer.Activation('sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='nadam')
    
    
    
    return model  


# In[49]:


early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=4, mode='auto')


# In[ ]:


model = get_model()


# In[ ]:


model.fit(X_train, y=y_train, 
                     validation_data = (X_val, y_val),
                     batch_size=4096, epochs=10000,
                     verbose=1, shuffle=True, callbacks=[early_stopping])


# In[ ]:


y_pred = model.predict(X[len(df_train):])
df_test = pd.read_csv('../input/test.csv')
df_test['deal_probability'] = y_pred.T[0]
df_test = df_test[['item_id','deal_probability']]


# In[ ]:


df_test.to_csv('word2score.csv', index=None)

