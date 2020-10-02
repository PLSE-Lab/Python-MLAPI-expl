#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re 
import math
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


data = pd.read_csv('../input/facebook-antivaccine-post-data-scaled-features/features_scaled.csv', 
                  index_col=0)
text = pd.read_csv('../input/facebook-antivaccination-dataset/posts_full.csv', 
                   index_col=0).text
assert text.shape[0] == data.shape[0]
data['text'] = text
del text
data.head()


# ## Prepare Data

# In[3]:


#Remove unwanted punctuation
unwanted = {x for x in '"$%&()*+,.!?-/:;<=>[\\]^_`{|}~\t\n'}

def filter_unwanted(x):
    x = "".join([c if c not in unwanted else " " for c in x]).lower()
    return x.encode("utf8").decode("ascii",'ignore')
data['text'] = [sentence for sentence in data.text.fillna('').apply(filter_unwanted)]
data.text.head()


# In[4]:


#Add n-gram input sequences
NUM_WORDS = 50_000
MAX_SEQUENCE_LENGTH = 200

tokenizer = Tokenizer(num_words=NUM_WORDS, filters='!"$%&()*+,-./:;<=>?[\\]^_`{|}~\t\n', 
                      lower=True)
tokenizer.fit_on_texts(data.text)

X1 = tokenizer.texts_to_sequences(data.text)
X1 = pad_sequences(X1, maxlen=MAX_SEQUENCE_LENGTH)
X1[-40:], data.text[0]


# In[22]:


X2 = data.drop(['anti_vax', 'text'], axis=1).values


# In[23]:


y = data.anti_vax.values


# In[24]:


X1_train, X1_eval, X2_train, X2_eval, y_train, y_eval = train_test_split(X1, X2, y, test_size=0.15, 
                                                                         random_state=3000)


# ## Build Model

# In[29]:


input1 = Input(shape=(X1.shape[1],), name="Text")
input2 = Input(shape=(X2.shape[1],), name="Text_Features")

#RNN of Text data
text_branch = Embedding(NUM_WORDS, 10)(input1)
text_branch = LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(text_branch)
text_branch = Dropout(rate=0.2)(text_branch)
text_branch = LSTM(64, dropout=0.2, recurrent_dropout=0.2)(text_branch)
aux_output = Dense(1, activation='sigmoid', name='Aux_Output')(text_branch)

#Text Features
text_feat = BatchNormalization()(input2)

#Join branches
x = concatenate([text_branch, text_feat])
main_branch = Dense(80, activation='relu')(x)
main_output = Dense(1, activation='sigmoid', name="Main_Output")(main_branch)

#Model
model = Model(inputs=[input1, input2], outputs=[main_output, aux_output])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'], 
              loss_weights={'Main_Output': 1.0, 'Aux_Output':0.2})
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)


# <h1 style="text-align:center">Final Model</h1>
# <img src="model.png" width="700">

# ## Train

# In[27]:


checkpoint = ModelCheckpoint("model-{epoch:02d}-{val_Main_Output_loss:.2f}.hdf5", 
                             monitor='val_Main_Output_loss', verbose=0, 
                             save_best_only=True, period=1)
stopping = EarlyStopping(monitor='val_Main_Output_loss', patience=5)
history = model.fit({'Text': X1_train, 'Text_Features': X2_train}, [y_train, y_train], epochs=20, 
                    verbose=2, batch_size=32, validation_data=([X1_eval, X2_eval], [y_eval, y_eval]), 
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


# In[ ]:




