#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

import tensorflow as tf
import keras

tsv_file = '../input/movie-review-sentiment-analysis-kernels-only/train.tsv'
train_data = pd.read_table(tsv_file)


# In[ ]:


train_data = train_data.values


# In[ ]:


sentiments = []
features = []

for i in range(0, len(train_data)):
    sentiments.append(train_data[i][3])
    
for i in range(0, len(train_data)):
    features.append(train_data[i][2])
    
sentences = features


# In[ ]:


max_features = 16467
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 55
batch_size = 32


# In[ ]:


from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

sentiments = to_categorical(sentiments)


# In[ ]:


from numpy import asarray
import os

embeddings_index = {}
f = open(os.path.join('../input/glove6b100dtxt', 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# In[ ]:


word_index = tokenizer.word_index

embedding_matrix = np.zeros((len(word_index) + 1, 100))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# In[ ]:


train_features = sequences[:156060]
train_targets = sentiments[:156060]

val_features = sequences[134848:]
val_targets = sentiments[134848:]


# In[ ]:


from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Embedding, GRU, Bidirectional


# In[ ]:


print('Pad sequences (samples x time)')
train_features = sequence.pad_sequences(train_features, maxlen=maxlen, padding='pre')
print('train_features shape:', train_features.shape)
train_features = np.array(train_features)


# In[ ]:


print('Pad sequences (samples x time)')
val_features = sequence.pad_sequences(val_features, maxlen=maxlen, padding='pre')
print('val_features shape:', val_features.shape)
val_features = np.array(val_features)


# In[ ]:


from keras.models import *
from keras.layers import *
from keras.initializers import *
from keras.optimizers import *


# In[ ]:


maxlen = 55
max_features = 15289
embed_size = 100


# In[ ]:


# https://www.kaggle.com/suicaokhoailang/lstm-attention-baseline-0-652-lb

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

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
        return input_shape[0],  self.features_dim


# In[ ]:


def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale

# A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(tf.keras.backend.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = tf.keras.backend.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


# In[ ]:


def BiGRUCapsNet():
    inp = Input(shape=(maxlen,))
    embedding = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    spatial_dropout = SpatialDropout1D(rate=0.2)(embedding)
    
    bi_gru_1 = Bidirectional(CuDNNGRU(64, return_sequences=True, 
                             kernel_initializer=glorot_normal(seed=12300), recurrent_initializer=orthogonal(gain=1.0, seed=10000)))(spatial_dropout)
    
    bi_gru_2 = Bidirectional(CuDNNGRU(64, return_sequences=True, 
                             kernel_initializer=glorot_normal(seed=12300), recurrent_initializer=orthogonal(gain=1.0, seed=10000)))(bi_gru_1)

    capsule = Capsule(num_capsule=10, dim_capsule=10, routings=4, share_weights=True)(bi_gru_1)
    capsule = Flatten()(capsule)
    
    gru_atten_1 = Attention(maxlen)(bi_gru_1)
    gru_atten_2 = Attention(maxlen)(bi_gru_2)

    avg_pool = GlobalAveragePooling1D()(bi_gru_2)
    max_pool = GlobalMaxPooling1D()(bi_gru_2)
    
    features = concatenate([gru_atten_1, gru_atten_2, capsule, avg_pool, max_pool])
    outp = Dense(5, activation='softmax')(features)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['acc'])
    return model


# In[ ]:


model = BiGRUCapsNet()
model.summary()


# In[ ]:


model.fit(x=train_features, y=train_targets, epochs=20, batch_size=128, validation_data=(val_features, val_targets))


# In[ ]:


tsv_file = '../input/movie-review-sentiment-analysis-kernels-only/test.tsv'
test_data = pd.read_table(tsv_file)


# In[ ]:


test_data = test_data.values


# In[ ]:


features = []

for i in range(0, len(test_data)):
    features.append(test_data[i][2])


# In[ ]:


sentences = features


# In[ ]:


from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

test_sequences = tokenizer.texts_to_sequences(sentences)
test_sequences = sequence.pad_sequences(test_sequences, maxlen=maxlen)


# In[ ]:


predictions = model.predict(np.array(test_sequences))

class_predictions = []

for i in range(0, len(predictions)):
    class_predictions.append(list.index(list(predictions[i]), max(predictions[i])))
    
class_predictions


# In[ ]:


ids = list(test_data[:, 0])


# In[ ]:


submission = pd.DataFrame(np.transpose(np.array([ids, class_predictions])))


# In[ ]:


submission.columns = ['PhraseId', 'Sentiment']


# In[ ]:


submission.to_csv('Movie-Review-Sentiment-Predictions-1.csv', index=False)


# In[ ]:




