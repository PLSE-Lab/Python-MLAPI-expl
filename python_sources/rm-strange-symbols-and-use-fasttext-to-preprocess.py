#!/usr/bin/env python
# coding: utf-8

# - Plan to train each column separately.
# - Here, use an rnn model to train the first column: `toxic`
# - It is pretty slow

# In[ ]:


## system
import os

## Math and dataFrame
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix, hstack

## Visualization
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from IPython.display import display
import seaborn as sns


# In[ ]:


## Traditional Machine Learning
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer


# In[ ]:


## Keras
from keras.layers import Input, Dropout, Dense, BatchNormalization, Activation, concatenate, GRU, Embedding, Flatten
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping#, TensorBoard
from keras import backend as K
from keras import optimizers
from keras.optimizers import SGD
from keras import initializers
from keras.callbacks import LearningRateScheduler
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences


# In[ ]:


## Using Multi processing


# In[ ]:


from multiprocessing import Pool


# In[ ]:


## Load data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# ##### Do some statistics

# In[ ]:


display(train[:10])
print(train.shape)
print("toxic count = {0}".format(train.toxic.sum()))
print("severe_toxic count = {0}".format(train.severe_toxic.sum()))
print("obscene count = {0}".format(train.obscene.sum()))
print("threat count = {0}".format(train.threat.sum()))
print("insult count = {0}".format(train.insult.sum()))
print("identity_hate count = {0}".format(train.identity_hate.sum()))


# ###### Show correlation matrix

# In[ ]:


corr = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].corr()
f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# ###### For Sentence processing

# In[ ]:


from nltk import word_tokenize
from nltk.corpus import stopwords
import string
stop = set(stopwords.words('english'))
punc = set(string.punctuation)


# Here I am trying to remove all the stop words and punctuations. Not sure whether it will give a better result or not

# In[ ]:


#### Preprocess sentences (removing punctuations and removing stop words)
def rmPunc(sent):
    return ''.join([ch for ch in str(sent) if ch not in punc])
def rmStop(sent):
    return ' '.join([word for word in sent.split() if word not in stop])


# In[ ]:


print("PREPROCESS TEXT...")
pool = Pool()
get_ipython().run_line_magic('time', 'train.comment_text = pool.map(rmPunc, train.comment_text.str.lower())')
get_ipython().run_line_magic('time', 'test.comment_text = pool.map(rmPunc, test.comment_text.str.lower())')

get_ipython().run_line_magic('time', 'train.comment_text = pool.map(rmStop, train.comment_text.str.lower())')
get_ipython().run_line_magic('time', 'test.comment_text = pool.map(rmStop, test.comment_text.str.lower())')
pool.close()
pool.join()


# ###### Tokenization

# In[ ]:


#PROCESS TEXT: RAW
print("Text to seq process...")
print("   Fitting tokenizer...")
from keras.preprocessing.text import Tokenizer
raw_text = np.hstack([train.comment_text.str.lower(), 
                      test.comment_text.str.lower()])
tok_raw = Tokenizer()
tok_raw.fit_on_texts(raw_text)
print("   Transforming text to seq...")
train["input"] = tok_raw.texts_to_sequences(train.comment_text.str.lower())
test["input"] = tok_raw.texts_to_sequences(test.comment_text.str.lower())


# ###### Some statistics on sentence lengths

# In[ ]:


test.input.apply(lambda x: len(x)).hist()
train.input.apply(lambda x: len(x)).hist()


# In[ ]:


MAX_LENGTH = 200
MAX_TOKEN = np.max([np.max(train.input.max()),np.max(test.input.max())]) + 5
print(MAX_LENGTH, MAX_TOKEN)


# In[ ]:


train = train[['input', 'toxic']]
dtrain, dvalid = train_test_split(train, random_state=17, train_size=0.7)
print(dtrain.shape)
print(dvalid.shape)


# ###### Artificially balance the classes

# In[ ]:


L = len(dtrain)
df_irr = dtrain[dtrain.toxic != 0]
while len(dtrain) < 2*L:
    dtrain = dtrain.append(df_irr, ignore_index=True)


# In[ ]:


L = len(dvalid)
df_irr = dvalid[dvalid.toxic != 0]
while len(dvalid) < 2*L:
    dvalid = dvalid.append(df_irr, ignore_index=True)


# ###### Creating RNN model

# In[ ]:


A = Input(shape=[MAX_LENGTH], name="in")
B = Embedding(MAX_TOKEN, 128)(A)
C = GRU(32) (B)
D = Dropout(0.6) (Dense(128, activation='relu') (C))
E = Dropout(0.4) (Dense(32, activation='relu') (D))
output = Dense(2, activation="softmax") (E)


# In[ ]:


model = Model(A, output)
N_epoch = 1
learning_rate = 0.05
optimizer = SGD(learning_rate)
loss = 'categorical_crossentropy'
metrics = ['accuracy']
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.summary()


# In[ ]:


train_x = pad_sequences(dtrain.input, maxlen=MAX_LENGTH)
valid_x = pad_sequences(dvalid.input, maxlen=MAX_LENGTH)
train_y = np_utils.to_categorical(dtrain.toxic.values, 2)
valid_y = np_utils.to_categorical(dvalid.toxic.values, 2)


# In[ ]:


res = model.fit(train_x, train_y, batch_size = 128, epochs = N_epoch, 
                verbose = 1, validation_data = (valid_x, valid_y))


# In[ ]:




