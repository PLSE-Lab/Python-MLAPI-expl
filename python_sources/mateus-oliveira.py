#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, SpatialDropout1D, Input, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from sklearn.metrics import roc_auc_score
import os
print(os.listdir("../input"))


# # Reading data

# In[34]:


df_train = pd.read_csv('../input/train.csv')
df_valid = pd.read_csv('../input/valid.csv')


# In[35]:


submission = pd.DataFrame()
submission['ID'] = df_valid['ID']


# In[36]:


df = pd.concat([df_train,df_valid],sort=True)

print(len(df_valid))
print(len(df_train))
print(len(df))
df.sample(5)


# In[37]:


df.isnull().sum()


# In[38]:


df = df[['headline','is_sarcastic']]
df.head()


# # Vectorizing inputs

# Vectorizing a text corpus, by turning each text into either a sequence of integers (each integer being the index of a token in a dictionary) or into a vector where the coefficient for each token could be binary, based on word count, based on tf-idf...
# 
# 

# In[39]:


tokenizer = Tokenizer(num_words=2000, split=' ')
tokenizer.fit_on_texts(df['headline'].values)
X = tokenizer.texts_to_sequences(df['headline'].values)
X = pad_sequences(X)


# # Train/Test split

# In[40]:


df_train = df[df['is_sarcastic'] >= 0]

y = pd.get_dummies(df_train['is_sarcastic']).values

X_train = X[:18696]
X_sub = X[18696:]

x_train, x_test, y_train, y_test = train_test_split(X_train, y, random_state=42, test_size=0.2)


# # Model

# ![LSTM](https://i.imgur.com/gl6EBCn.png)

# *dropout*: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.
# 
# *recurrent_dropout*: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state.

# In[42]:


model = Sequential([
    Embedding(2000, 200, input_length = X.shape[1]),
    SpatialDropout1D(0.2),
    LSTM(200, dropout=0.2, recurrent_dropout=0.2),
    Dense(2,activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# ## Model training/Evaluation

# In[43]:


model.fit(x_train, y_train, epochs = 25, verbose = 2)


# In[44]:


valid_pred = model.predict(x_test)

print("AUC: %.2f" % roc_auc_score(y_test, valid_pred))


# # Generating Submission

# In[45]:


model.fit(X_train, y, epochs = 200, verbose = 2)


# In[46]:


pred = model.predict(X_sub,batch_size=1,verbose = 0)

submission['is_sarcastic'] = np.argmax(pred, axis= 1)

submission.head()


# In[47]:


submission.to_csv('submission.csv', index = False)

