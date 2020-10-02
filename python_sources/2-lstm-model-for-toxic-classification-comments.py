#!/usr/bin/env python
# coding: utf-8

# # Bi-LSTM

# Natural Language Processing(NLP) is one of the main usages for the Neural networks-Deep learning model wither it was speech recognition, ChatBots or even predict the next words in a sentence, this all will not be achieved throughout using simple NN there is model's developed in order to overcome these obstacles one of these models is RNN.
# 

# BiLSTM - (Bidirectional LSTMs) it's an extension of traditional LSTMs. It trains two instead of one LSTMs on the input sequence, The first on the input sequence as-is, and the second on a reversed copy of the input sequence. This can provide additional context to the network and result in faster and even fuller learning on the problem.

# Note: This file is created to experment the run for explined jupter file 

# ## The Code

# In[ ]:


from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from numba import jit, cuda 
from timeit import default_timer as timer    

import os,pickle
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation,Layer
from keras.layers import Bidirectional,GlobalAveragePooling1D#Concatenate
from keras.models import Model,Sequential
from sklearn import metrics
from keras.preprocessing import sequence, text 
from keras import initializers, regularizers, constraints, optimizers, layers # This for the attenition layer
from tensorflow.keras import backend as K #this is also for the attention layer
import tensorflow as tf
import transformers
from keras.layers.merge import concatenate
from tensorflow.keras import layers


# ## Loading Data Sets

# In[ ]:


# Loading train sets
train = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")

# Loading validation sets
valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')

# Loading test sets
test = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/test.csv')
sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')


# ## DATA Pre-processing

# In[ ]:


# select comment_text for the preprosess (X)
list_sentences_train = train["comment_text"] 
list_sentences_validation = valid['comment_text']
list_sentences_test = test["content"]

#select comment_text for the preprosess (y)
y_train = train.toxic.values 
y_valid = valid.toxic.values 


# In[ ]:


# call the tokenizer with it's paramitera
max_features = 20000
tokenizer = Tokenizer(num_words=max_features,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True) 

#Fitting tokenizer
tokenizer.fit_on_texts(list(list_sentences_train) + list(list_sentences_validation) + list(list_sentences_test))

# for bulding the matrix
word_index = tokenizer.word_index

# Building training set
list_tokenized_train = tokenizer.texts_to_sequences(list(list_sentences_train))
y_train = train['toxic'].values

# Building validation set
list_tokenized_validation = tokenizer.texts_to_sequences(list(list_sentences_validation))
y_valid = valid['toxic'].values

# Building test set
list_tokenized_test = tokenizer.texts_to_sequences(list(list_sentences_test))

del tokenizer # To save RAM space


# In[ ]:


maxlen = 200 # length of padding

# Padding sequences for all 
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_valid = pad_sequences(list_tokenized_validation, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)


# In[ ]:


# using Crawl word vector
with open('../input/pickled-crawl300d2m-for-kernel-competitions/crawl-300d-2M.pkl', 'rb') as  infile:
        crawl_embeddings = pickle.load(infile)


# In[ ]:


# using GloVe word vector
with open('../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl', 'rb') as  infile:
        glove_embeddings = pickle.load(infile)


# In[ ]:


# function for building a matrix
def build_matrix(word_index, embeddings_index):
    ''''
    Input: word indexing from the tocnizer appove and the pre-trined word vector model
    
    output: embedding matrix
    
    ''''
    embedding_matrix = np.zeros((len(word_index) + 1,300 ))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embeddings_index[word]
        except:
            embedding_matrix[i] = embeddings_index["unknown"]
    return embedding_matrix


# In[ ]:


# Building matrices
embedding_matrix_1 = build_matrix(word_index, crawl_embeddings)
embedding_matrix_2 = build_matrix(word_index, glove_embeddings)

# Concatenating embedding matrices 
embedding_matrix = np.concatenate([embedding_matrix_1, embedding_matrix_2], axis=1)

del embedding_matrix_1, embedding_matrix_2
del crawl_embeddings ,glove_embeddings  # for saving RAM Space


# ## Modeling

# In[ ]:


class Attention(Layer):
    """
    Custom Keras attention layer
    
    Reference: https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043
    """
    def __init__(self, step_dim, W_regularizer=None, b_regularizer=None, 
                 W_constraint=None, b_constraint=None, bias=True, **kwargs):

        self.supports_masking = True

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = None
        super(Attention, self).__init__(**kwargs)

        self.param_W = {
            'initializer': initializers.get('glorot_uniform'),
            'name': '{}_W'.format(self.name),
            'regularizer': regularizers.get(W_regularizer),
            'constraint': constraints.get(W_constraint)
        }
        self.W = None

        self.param_b = {
            'initializer': 'zero',
            'name': '{}_b'.format(self.name),
            'regularizer': regularizers.get(b_regularizer),
            'constraint': constraints.get(b_constraint)
        }
        self.b = None

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.features_dim = input_shape[-1]
        self.W = self.add_weight(shape=(input_shape[-1],), 
                                 **self.param_W)

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],), 
                                     **self.param_b)

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        step_dim = self.step_dim
        features_dim = self.features_dim

        eij = K.reshape(
            K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))),
            (-1, step_dim))

        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


# In[ ]:


#define shape of the input 
inp = Input(shape=(maxlen,)) 


# In[ ]:


# create embedding layer 
embedding_layer = Embedding(*embedding_matrix.shape,
                                weights=[embedding_matrix],
                                trainable=False) 


# In[ ]:


# pass input into the embded lyer 
x = embedding_layer(inp) 


# In[ ]:


# feed into bidirectional wech it will out but 
x = Bidirectional(LSTM(256, return_sequences=True))(x) 


# In[ ]:


# feed into bidirectional wech it will out but
x = Bidirectional(LSTM(128, return_sequences=True))(x) 


# In[ ]:


# call the GlobalAveragePooling1D 
avrege = GlobalAveragePooling1D()(x)


# In[ ]:


# call the Attention 
attention = Attention(maxlen)(x)


# In[ ]:


# concate these techniqes to form layer that perform on the output from the Bi-LSTM 
hidden = concatenate([attention,avrege])


# In[ ]:


# using dense with 512 output with relu acttivation function
x = Dense(512, activation='relu')(hidden)


# In[ ]:


# perform a dropout with 0.5 to avoid ofer fitting 
x =  Dropout(0.5)(x)


# In[ ]:


# using dense with 128 output with relu acttivation function 
x = Dense(128, activation="relu")(x)


# In[ ]:


# using dense output with sigmoid acttivation function 
o = Dense(1, activation='sigmoid')(x)


# In[ ]:


# call the model 
model = Model(inputs=inp, outputs=o)
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[tf.keras.metrics.AUC()])


# In[ ]:


# Model fitting on train data set
model.fit(X_t,y_train,batch_size=32,epochs=2,validation_split=0.1)

# NOTE : THE RUN MAY TAKE SOME TIME 


# In[ ]:


# Model fitting on Validation data set
model.fit(X_valid,y_valid,batch_size=32,epochs=2,validation_split=0.1)
# NOTE : THE RUN MAY TAKE SOME TIME 


# In[ ]:


# Predect the toxicity of the test
val = model.predict(X_te, verbose=1)


# In[ ]:


# save the predections into the submetion file 
sub['toxic'] = val 
sub.to_csv('submission.csv', index=False)

