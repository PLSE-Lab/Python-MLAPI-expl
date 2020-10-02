#!/usr/bin/env python
# coding: utf-8

# In[22]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import gc
from keras import backend as K
from keras.layers import Dense,Input,Bidirectional,Activation,Conv1D,GRU
from keras.callbacks import Callback
from keras.layers import Dropout,Embedding,GlobalMaxPooling1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
# Any results you write to the current directory are saved as output.


# In[ ]:


EMBEDDING_FILE = '../input/fasttext-russian-2m/wiki.ru.vec'

train = pd.read_csv('../input/avito-demand-prediction/train.csv')
test = pd.read_csv('../input/avito-demand-prediction/test.csv')


# In[ ]:


train['description'] = train['description'].astype(str)
test["description"] = test['description'].astype(str)


# In[ ]:


train["description"] = train["description"].fillna("fillna")
test["description"] = test["description"].fillna("fillna")


# In[ ]:


train['description'] = (train['description'].fillna('') + ' ' + train['title'].fillna('') + ' ' + train['parent_category_name'].fillna('') + ' ' + train['category_name'].fillna('') + ' ' + train['param_1'].fillna(''))
test['description'] = (test['description'].fillna('') + ' ' + test['title'].fillna('') + ' ' + test['parent_category_name'].fillna('') + ' ' + test['category_name'].fillna('') + ' ' + test['param_1'].fillna(''))


# In[10]:


X_train = train["description"]
y_train = train[["deal_probability"]].values
del train
gc.collect()

X_test = test["description"]
del test
gc.collect()


# In[11]:


max_features=50000
maxlen=100
embed_size=300


# In[12]:


from keras.preprocessing import text, sequence
tok=text.Tokenizer(num_words=max_features)
tok.fit_on_texts(X_train)
X_train=tok.texts_to_sequences(X_train)
X_test=tok.texts_to_sequences(X_test)
x_train=sequence.pad_sequences(X_train,maxlen=maxlen)
del X_train
gc.collect()
x_test=sequence.pad_sequences(X_test,maxlen=maxlen)
del X_test
gc.collect()


# In[13]:


embeddings_index = {}
with open(EMBEDDING_FILE,encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float16')
        embeddings_index[word] = coefs


# In[14]:


word_index = tok.word_index
#prepare embedding matrix
num_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# In[15]:


del tok
gc.collect()


# Defining Capsule Network

# In[16]:


from keras.layers import K, Activation
from keras.engine import Layer
from keras.layers import Dense, Input, Embedding, Dropout, Bidirectional, GRU, Flatten, SpatialDropout1D, CuDNNLSTM

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
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


# In[17]:


Routings = 6
Num_capsule = 10
Dim_capsule = 16
rate_drop_dense = 0.35

#def root_mean_squared_error(y_true, y_pred):
#    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 
sequence_input = Input(shape=(maxlen, ))
x = Embedding(max_features, embed_size, weights=[embedding_matrix],trainable = False)(sequence_input)
x = SpatialDropout1D(0.2)(x)
x = Bidirectional(GRU(32, return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x)
capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                      share_weights=True)(x)
capsule = Flatten()(capsule)
capsule = Dropout(0.4)(capsule)
preds = Dense(1, activation="sigmoid")(capsule)
model = Model(sequence_input, preds)
model.compile(loss='MSE',optimizer=Adam(lr=1e-3),metrics=['accuracy', root_mean_squared_error])


# In[18]:


batch_size = 1000
epochs = 3
X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.9, random_state=233)
del x_train
gc.collect()
del y_train
gc.collect()


# In[ ]:


# filepath="../input/best-model/best.hdf5"
filepath="weights_base.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_root_mean_squared_error', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_root_mean_squared_error", mode="min", patience=5)
callbacks_list = [checkpoint, early]


# In[ ]:


model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),callbacks = callbacks_list,verbose=1)
#Loading model weights
model.load_weights(filepath)
print('Predicting....')
y_pred = model.predict(x_test,batch_size=1024,verbose=1)


# In[ ]:


sub = pd.read_csv('../input/avito-demand-prediction/sample_submission.csv')
sub['deal_probability'] = y_pred
sub['deal_probability'].clip(0.0, 1.0, inplace=True)
sub.to_csv('gru_capsule_description.csv', index=False)


# In[ ]:





# In[ ]:




