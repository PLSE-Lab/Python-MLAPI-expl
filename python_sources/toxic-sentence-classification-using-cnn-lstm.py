#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from bs4 import BeautifulSoup             
from nltk.corpus import stopwords # Import the stop word list
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU,Conv1D,MaxPooling1D
from keras.layers import Bidirectional, GlobalMaxPool1D,Bidirectional
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import gc
from sklearn.model_selection import train_test_split
from keras.models import load_model


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
submit_template = pd.read_csv('../input/sample_submission.csv', header = 0)


# In[ ]:


train.head()


# In[ ]:


list_sentences = train["comment_text"]
list_sentences_test = test["comment_text"]


# In[ ]:


max_features = 20000
tokenizer = Tokenizer(num_words=max_features,char_level=True)


# In[ ]:


tokenizer.fit_on_texts(list(list_sentences))


# In[ ]:


list_tokenized = tokenizer.texts_to_sequences(list_sentences)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)


# In[ ]:


maxlen = 500
X_t = pad_sequences(list_tokenized, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)


# Just in case you are wondering, the reason why I used 500 is because most of the number of characters in a sentence falls within 0 to 500:

# In[ ]:


totalNumWords = [len(one_comment) for one_comment in list_tokenized]
plt.hist(totalNumWords)
plt.show()


# Finally, we can start buliding our model.
# 
# First, we set up our input layer. As mentioned in the Keras documentation, we have to include the shape for the very first layer and Keras will automatically derive the shape for the rest of the layers.

# In[ ]:


inp = Input(shape=(maxlen, ))
inp


# In[ ]:


embed_size = 240
x = Embedding(len(tokenizer.word_index)+1, embed_size)(inp)


# In[ ]:


x = Conv1D(filters=100,kernel_size=4,padding='same', activation='relu')(x)


# Then we pass it to the max pooling layer that applies the max pool operation on a window of every 4 characters. And that is why we get an output of (num of sentences X 125 X 100) matrix.

# In[ ]:


x=MaxPooling1D(pool_size=4)(x)


# In[ ]:


x = Bidirectional(GRU(60, return_sequences=True,name='lstm_layer',dropout=0.2,recurrent_dropout=0.2))(x)


# In[ ]:


x = GlobalMaxPool1D()(x)


# In[ ]:


x = Dense(50, activation="relu")(x)


# In[ ]:


x = Dropout(0.2)(x)
x = Dense(6, activation="sigmoid")(x)


# In[ ]:


model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                 metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_t, train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]], test_size = 0.10, random_state = 42)


# In[ ]:


batch_size = 32
epochs = 6
model.fit(X_train,y_train, batch_size=batch_size, epochs=epochs,validation_data=(X_test,y_test),verbose=2)


# In[ ]:


y_submit = model.predict(X_te,batch_size=batch_size,verbose=1)


# In[ ]:


y_submit[np.isnan(y_submit)]=0
sample_submission = submit_template
sample_submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_submit
sample_submission.to_csv('submission.csv', index=False)

