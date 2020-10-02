#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re 
import math
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical, Sequence, plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
from keras.layers import Embedding, Dense, Dropout, LSTM, Input, BatchNormalization, concatenate

from tensorflow import set_random_seed
from numpy.random import seed
set_random_seed(2)
seed(40)


# In[2]:


data = pd.read_csv('../input/facebook-antivaccination-dataset/posts_full.csv', 
                   index_col=0).dropna(subset=['text'])
data = data[['text', 'anti_vax']]
data.head()


# In[3]:


discord = pd.read_csv('../input/game-of-cones/Game of Cones - battle-for-the-cone 231969664027066368.csv', 
                      sep=';').drop('Unnamed: 4', axis=1).dropna(subset=['Content'])
discord['anti_vax'] = False
discord = discord[['Content', 'anti_vax']].rename({'Content': 'text'}, axis=1)
discord.head()


# ## Prepare Data

# In[4]:


data = pd.concat([data, discord]).reset_index(drop=True)
data.shape


# ## Build Tokenizer

# In[5]:


#Remove unwanted punctuation
FILTER_STRING = '"$%&()*+,.!?-/:;<=>[\\]@#^_`{|}~\t\n'
UNWANTED = {x for x in FILTER_STRING}
def filter_unwanted(x):
    x = "".join([c if c not in UNWANTED else " " for c in x]).lower()
    return x.encode("utf8").decode("ascii",'ignore')


# In[6]:


data['text'] = [sentence for sentence in data.text.apply(filter_unwanted)]
data.text.tail()


# In[7]:


#Add n-gram input sequences
NUM_WORDS = 50_000
MAX_SEQUENCE_LENGTH = 200

tokenizer = Tokenizer(num_words=NUM_WORDS, filters=FILTER_STRING, 
                      lower=True)
tokenizer.fit_on_texts(data.text)
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


# ## Build Training Sets

# In[8]:


X = tokenizer.texts_to_sequences(data.text)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
X[0][-40:], data.text.head(1)


# In[9]:


X_train, X_eval, y_train, y_eval = train_test_split(X, data.anti_vax.values, 
                                                    test_size=0.2, 
                                                    random_state=3000)


# ## Build Model

# In[11]:


model = Sequential()
model.add(Embedding(NUM_WORDS, 10, input_length=(X.shape[1])))
model.add(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
model.add(Dropout(rate=0.3))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)


# <h1 style="text-align:center">Final Model</h1>
# <img src="model.png" width="200">

# ## Train

# In[13]:


checkpoint = ModelCheckpoint("model-{epoch:02d}-{val_loss:.2f}.hdf5", 
                             monitor='val_loss', verbose=0, 
                             save_best_only=True, period=1)
stopping = EarlyStopping(monitor='val_loss', patience=2)
history = model.fit(X_train, y_train, epochs=20, 
                    verbose=2, batch_size=32, validation_data=(X_eval, y_eval), 
                    callbacks=[checkpoint, stopping])


# ## Comparing Loss per Epoch

# In[ ]:


def plot_epochs(results, col, **kwargs):
    def plot_epoch_helper(hist_df, col, ax):
        ax.plot(hist_df[col], **kwargs)
        ax.set_title(col + ' per epoch')
        ax.set_ylabel(col)
        ax.set_xlabel('epoch')
        for sp in ax.spines:
            ax.spines[sp].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3)
        ax.legend(labels=[n[0] for n in results])
        ax.set_ylim(0, 1)
    fig, ax = plt.subplots(figsize=(21, 10))
    for name, hist in results:
        plot_epoch_helper(hist, col, ax)
plot_epochs([('Model', pd.DataFrame(history.history))], 'val_Main_Output_loss')


# In[ ]:


plot_epochs([('Model', pd.DataFrame(history.history))], 'val_Aux_Output_loss')

