#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import sys
import numpy as np
import pandas as pd
import pickle
from pymagnitude import *
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import gc
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import RNN, GRU, LSTM, Dense, Input, Embedding, Dropout, Activation, concatenate
from keras.layers import Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.callbacks import Callback, ReduceLROnPlateau
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU
from keras.callbacks import Callback
from keras.layers import Dropout,Embedding,GlobalMaxPooling1D, MaxPooling1D, Add, Flatten
from keras.preprocessing import text, sequence
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import pandas as pd
import os
import nltk
import re
from bs4 import BeautifulSoup
import urllib3
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
from sklearn import preprocessing
from scipy import sparse
from keras import backend as K # Importing Keras backend (by default it is Tensorflow)
from keras.layers import Input, Dense # Layers to be used for building our model
from keras.models import Model # The class used to create a model
from keras.optimizers import Adam
from keras.utils import np_utils # Utilities to manipulate numpy arrays
from tensorflow import set_random_seed # Used for reproducible experiments
from tensorflow import keras
import gc
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.layers import InputLayer, Input, Embedding, Dense, Dropout, Bidirectional, GlobalMaxPool1D, GlobalAveragePooling1D, SpatialDropout1D, Conv1D, CuDNNLSTM, CuDNNGRU, TimeDistributed, Reshape, Permute, LocallyConnected1D, concatenate, ELU, Activation, add, Lambda, BatchNormalization, PReLU, MaxPooling1D, GlobalMaxPooling1D
from keras.optimizers import Adam
from keras import regularizers
#from kgutil.models.keras.base import DefaultTrainSequence, DefaultTestSequence
#from kgutil.models.keras.rnn import KerasRNN, load_emb_matrix
from copy import deepcopy
import inspect

import os


# ## Load Create the sequence embeddings

# In[ ]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train_data[classes].values

train_sentences = train_data["comment_text"].fillna("fillna").str.lower()
test_sentences = test_data["comment_text"].fillna("fillna").str.lower()

max_features = 100000
max_len = 150
embed_size = 300

tokenizer = Tokenizer(max_features)
tokenizer.fit_on_texts(list(train_sentences))

tokenized_train_sentences = tokenizer.texts_to_sequences(train_sentences)
tokenized_test_sentences = tokenizer.texts_to_sequences(test_sentences)

train_padding = pad_sequences(tokenized_train_sentences, max_len)
test_padding = pad_sequences(tokenized_test_sentences, max_len)

#max_len = 150
#https://github.com/plasticityai/magnitude
#!curl -s http://magnitude.plasticity.ai/glove+subword/glove.6B.300d.magnitude --output vectors.magnitude

#vecs_word2vec = Magnitude('http://magnitude.plasticity.ai/word2vec/heavy/GoogleNews-vectors-negative300.magnitude', stream=True, pad_to_length=max_len) 
vecs_glove = Magnitude('http://magnitude.plasticity.ai/glove+subword/glove.6B.300d.magnitude')
#vecs_fasttext = Magnitude('http://magnitude.plasticity.ai/fasttext+subword/wiki-news-300d-1M.magnitude', pad_to_length=max_len)
#vecs_elmo = Magnitude('http://magnitude.plasticity.ai/elmo/medium/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.magnitude', stream=True, pad_to_length=max_len)

#vectors = Magnitude(vecs_fasttext, vecs_glove) # concatenate word2vec with glove
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, vecs_glove.dim))

from tqdm import tqdm_notebook as tqdm
for word, i in tqdm(word_index.items()):
    if i >= max_features:
        continue
    embedding_vector = vecs_glove.query(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

gc.collect()


# ### GRU Autoencoder

# In[ ]:


from keras.layers import *

num_words = 100000
maxlen = 150
embed_dim = 300
latent_dim = 128
batch_size = 32
pad_seqs = train_padding

#### Encoder Model ####
encoder_inputs = Input(shape=(maxlen,), name='Encoder-Input')
emb_layer = Embedding(num_words, embed_dim,input_length = maxlen, name='Body-Word-Embedding', mask_zero=False)
# Word embeding for encoder (ex: Issue Body)
x = emb_layer(encoder_inputs)
state_h = GRU(latent_dim, name='Encoder-Last-GRU')(x)
encoder_model = Model(inputs=encoder_inputs, outputs=state_h, name='Encoder-Model')
seq2seq_encoder_out = encoder_model(encoder_inputs)
#### Decoder Model ####
decoded = RepeatVector(maxlen)(seq2seq_encoder_out)
decoder_gru = GRU(latent_dim, return_sequences=True, name='Decoder-GRU-before')
decoder_gru_output = decoder_gru(decoded)
decoder_dense = Dense(num_words, activation='softmax', name='Final-Output-Dense-before')
decoder_outputs = decoder_dense(decoder_gru_output)
#### Seq2Seq Model ####
#seq2seq_decoder_out = decoder_model([decoder_inputs, seq2seq_encoder_out])
seq2seq_Model = Model(encoder_inputs,decoder_outputs )
seq2seq_Model.compile(optimizer=optimizers.Nadam(lr=0.001), loss='sparse_categorical_crossentropy')

history = seq2seq_Model.fit(pad_seqs, np.expand_dims(pad_seqs, -1),
          batch_size=batch_size,
          epochs=4,
          validation_split=0.12)

model_file = "autoencoder.sav"
with open(model_file,mode='wb') as model_f:
    pickle.dump(seq2seq_Model,model_f)


# In[ ]:


headlines = encoder_model.predict(pad_seqs)


# In[ ]:


# https://www.kaggle.com/yekenot/pooled-gru-fasttext

#Define a class for model evaluation
class RocAucEvaluation(Callback):
    def __init__(self, training_data=(),validation_data=()):
        super(Callback, self).__init__()
       
        self.X_tra, self.y_tra = training_data
        self.X_val, self.y_val = validation_data
        self.aucs_val = []
        self.aucs_tra = []
        
    def on_epoch_end(self, epoch, logs={}):                   
        y_pred_val = self.model.predict(self.X_val, verbose=0)
        score_val = roc_auc_score(self.y_val, y_pred_val)

        y_pred_tra = self.model.predict(self.X_tra, verbose=0)
        score_tra = roc_auc_score(self.y_tra, y_pred_tra)

        self.aucs_tra.append(score_val)
        self.aucs_val.append(score_tra)
        print("\n ROC-AUC - epoch: %d - score_tra: %.6f - score_val: %.6f \n" % (epoch+1, score_tra, score_val))
        
class Plots:
    def plot_history(history):
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        x = range(1, len(val_loss) + 1)

        plt.plot(x, loss, 'b', label='Training loss')
        plt.plot(x, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

    def plot_roc_auc(train_roc, val_roc):
        x = range(1, len(val_roc) + 1)

        plt.plot(x, train_roc, 'b', label='Training RocAuc')
        plt.plot(x, val_roc, 'r', label='Validation RocAuc')
        plt.title('Training and validation RocAuc')
        plt.legend()


# In[ ]:


X_tra, X_val, y_tra, y_val = train_test_split(headlines, y, train_size=0.90, random_state=233)
RocAuc = RocAucEvaluation(training_data=(X_tra, y_tra) ,validation_data=(X_val, y_val))


# In[ ]:


def MLP_model(
    input_size,
    optimizer,    
    classes=6,  
    epochs=100,
    batch_size=128,
    hidden_layers=1,
    units=600,
    dropout_rate=0.8,
    l2_lambda=0.0,
    batch_norm=False,
    funnel=False,
    hidden_activation='relu',
    output_activation='sigmoid'
):
  
    # Define the seed for numpy and Tensorflow to have reproducible experiments.
    np.random.seed(1402) 
    set_random_seed(1981)
       
    input = Input(
        shape=(input_size,),
        name='Input'
    )
    x = input
    print(x.shape)
    # Define the hidden layers.
    for i in range(hidden_layers):
        if funnel:
            layer_units=units // (i+1)
        else: 
            layer_units=units
        x = Dense(
           units=layer_units,
           kernel_initializer='glorot_uniform',
           kernel_regularizer=l2(l2_lambda),
           activation=hidden_activation,
           name='Hidden-{0:d}'.format(i + 1)
        )(x)
        #Dropout
        if dropout_rate != 0:
            x = Dropout(dropout_rate)(x)
        if batch_norm:
            x = BatchNormalization()(x)
            
    # Define the output layer.    
    output = Dense(
        units=classes,
        kernel_initializer='uniform',
        activation=output_activation,
        name='Output'
    )(x)
    # Define the model and train it.
    model = Model(inputs=input, outputs=output)
      
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_crossentropy'])
    
    return model


# In[ ]:


batch_size = 256
num_classes = 6
epochs = 80
optimizer = Adam(lr=0.001)

model = MLP_model(
    input_size = X_tra.shape[1],
    optimizer = optimizer,    
    classes=num_classes,  
    epochs=100,
    batch_size=128,
    hidden_layers=2,
    units=100,
    dropout_rate=0.5,
    l2_lambda=0.0,
    batch_norm=False,
    funnel=True,
    hidden_activation='relu',
    output_activation='sigmoid'
)

# Keras Callbacks
reducer_lr = ReduceLROnPlateau(factor = 0.00002, patience = 1, min_lr = 1e-6, verbose = 1)
early_stopper = keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', mode='min', patience = 10) # Change 4 to 8 in the final run
model_file_name = 'weights_base.best.hdf5'
check_pointer = keras.callbacks.ModelCheckpoint(model_file_name, monitor='val_binary_crossentropy', mode='min', verbose = 1, save_best_only = True)  
log_file_name = 'model.log'
csv_logger = keras.callbacks.CSVLogger(log_file_name)
# replaydata = ReplayData(X_tra, y_tra, filename='hyperparms_in_action.h5', group_name='part1')
callbacks_list = [early_stopper, check_pointer, csv_logger, RocAuc, reducer_lr]

model.fit(x=X_tra,
          y=y_tra,          
          validation_data=(X_val, y_val),
          epochs=epochs,
          shuffle=True,
          verbose=1,
          batch_size=batch_size,
          callbacks = callbacks_list)

print('Finished training.')
print('------------------')


# In[ ]:


# model.summary() # Print a description of the model.
# Plots.plot_roc_auc(RocAuc.aucs_tra, RocAuc.aucs_val)
# Plots.plot_history(model.history)


# In[ ]:


K.clear_session()
del seq2seq_Model
gc.collect()

