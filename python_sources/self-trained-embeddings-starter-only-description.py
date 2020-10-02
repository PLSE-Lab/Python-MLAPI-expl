#!/usr/bin/env python
# coding: utf-8

# ## Using self-trained embeddings from train_active on the description
# 
# This kernel shows how to use the Word2Vec model created in [this kernel](https://www.kaggle.com/christofhenkel/using-train-active-for-training-word-embeddings) on the description. To compare the performance with [the pre-trained embedding model](https://www.kaggle.com/christofhenkel/fasttext-starter-description-only) we use exactly the same model structure.

# In[1]:


import pandas as pd
from gensim.models import word2vec
from keras.preprocessing import text, sequence
import numpy as np
from tqdm import tqdm
from keras.layers import Input, SpatialDropout1D,Dropout, GlobalAveragePooling1D, CuDNNGRU, Bidirectional, Dense, Embedding
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import os


EMBEDDING = '../input/using-train-active-for-training-word-embeddings/avito.w2v'
TRAIN_CSV = '../input/avito-demand-prediction/train.csv'
TEST_CSV = '../input/avito-demand-prediction/test.csv'


# In[2]:


max_features = 100000
maxlen = 100
embed_size = 100
train = pd.read_csv(TRAIN_CSV, index_col = 0)
labels = train[['deal_probability']].copy()
train = train[['description']].copy()

tokenizer = text.Tokenizer(num_words=max_features)

print('fitting tokenizer...',end='')
train['description'] = train['description'].astype(str)
tokenizer.fit_on_texts(list(train['description'].fillna('NA').values))
print('done.')


# In[3]:


model = word2vec.Word2Vec.load(EMBEDDING)
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    try:
        embedding_vector = model[word]
    except KeyError:
        embedding_vector = None
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# In[4]:


X_train, X_valid, y_train, y_valid = train_test_split(train['description'].values, labels['deal_probability'].values, test_size = 0.1, random_state = 23)

print('convert to sequences...',end='')
X_train = tokenizer.texts_to_sequences(X_train)
X_valid = tokenizer.texts_to_sequences(X_valid)
print('done.')
print('padding...',end='')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_valid = sequence.pad_sequences(X_valid, maxlen=maxlen)
print('done.')

del train


# Lets define the model and illustrate the architecture

# In[7]:


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def build_model():
    inp = Input(shape = (maxlen, ))
    emb = Embedding(nb_words, embed_size, weights = [embedding_matrix],
                    input_length = maxlen, trainable = False)(inp)
    main = SpatialDropout1D(0.2)(emb)
    main = Bidirectional(CuDNNGRU(32,return_sequences = True))(main)
    main = GlobalAveragePooling1D()(main)
    main = Dropout(0.2)(main)
    out = Dense(1, activation = "sigmoid")(main)

    model = Model(inputs = inp, outputs = out)

    model.compile(optimizer = Adam(lr=0.001), loss = 'mean_squared_error',
                  metrics =[root_mean_squared_error])
    
    return model

model = build_model()
model.summary()


# Lets train our model for four epochs and save the best epoch.

# In[8]:


EPOCHS = 4
file_path = "model.hdf5"

check_point = ModelCheckpoint(file_path, monitor = "val_loss", mode = "min", save_best_only = True, verbose = 1)
history = model.fit(X_train, y_train, batch_size = 256, epochs = EPOCHS, validation_data = (X_valid, y_valid),
                verbose = 1, callbacks = [check_point])


# In[9]:


model.load_weights(file_path)
prediction = model.predict(X_valid)
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, prediction)))


# Thats some improvement compared to using  [the pre-trained embedding model](https://www.kaggle.com/christofhenkel/fasttext-starter-description-only) which scored 0.2370. Additionally since the embeddings here are trained also on param_1, param_2, param_3 and title which have much more out of vocabulary words when using Fasttext. Hence self-trained embeddings are clearly performing better.

# Ok, now we are ready to do a submission and compare the LB score with [the pre-trained embedding model](https://www.kaggle.com/christofhenkel/fasttext-starter-description-only).

# In[ ]:


test = pd.read_csv(TEST_CSV, index_col = 0)
test = test[['description']].copy()

test['description'] = test['description'].astype(str)
X_test = test['description'].values
X_test = tokenizer.texts_to_sequences(X_test)

print('padding')
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
prediction = model.predict(X_test,batch_size = 128, verbose = 1)

sample_submission = pd.read_csv('../input/avito-demand-prediction/sample_submission.csv', index_col = 0)
submission = sample_submission.copy()
submission['deal_probability'] = prediction
submission.to_csv('submission.csv')


# In[ ]:




