#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm


# In[ ]:


train = pd.read_csv('../input/train.csv')
train.head()


# In[ ]:


cols = list(train.columns)[2:]


# In[ ]:


from sklearn.preprocessing import StandardScaler

from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Dropout, Flatten, Input
from keras import backend as K
import keras
from matplotlib.colors import LogNorm


# In[ ]:


get_ipython().run_cell_magic('time', '', 'ss = StandardScaler(copy=False)\ndata_ss = ss.fit_transform(np.nan_to_num(train[cols].apply(lambda x: round(x, 2)).values))')


# # Make NN and train it

# In[ ]:


n_features = data_ss.shape[1]

dim = 15

def build_model(dropout_rate=0.15, activation='tanh'):
    main_input = Input(shape=(n_features, ), name='main_input')
    
    x = Dense(dim*2, activation=activation)(main_input)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(dim*2, activation=activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate/2)(x)
    
    x = Dense(dim, activation=activation)(x)
    x = Dropout(dropout_rate/4)(x)

    encoded = Dense(2, activation='tanh')(x)

    input_encoded = Input(shape=(2, ))
    
    x = Dense(dim, activation=activation)(input_encoded)
    x = Dense(dim, activation=activation)(x)
    x = Dense(dim*2, activation=activation)(x)
    
    decoded = x = Dense(n_features, activation='linear')(x)

    encoder = Model(main_input, encoded, name="encoder")
    decoder = Model(input_encoded, decoded, name="decoder")
    autoencoder = Model(main_input, decoder(encoder(main_input)), name="autoencoder")
    return encoder, decoder, autoencoder

K.clear_session()
c_encoder, c_decoder, c_autoencoder = build_model()
c_autoencoder.compile(optimizer='nadam', loss='mse')

c_autoencoder.summary()


# In[ ]:


get_ipython().run_cell_magic('time', '', "loss_history = []\nfor i in tqdm(range(20)):\n    epochs = 20\n    batch_size = 2048\n    history = c_autoencoder.fit(data_ss + np.random.normal(scale=0.01, size=data_ss.shape), data_ss,\n                        epochs=epochs,\n                        batch_size=batch_size,\n                            shuffle=True,\n                            verbose=0)\n    \n    loss_history += history.history['loss']\nplt.figure(figsize=(10, 5))\nplt.plot(loss_history);")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'emb = c_encoder.predict(data_ss)')


# # Draw result
# ## Density

# In[ ]:


plt.figure(figsize=(10, 10))
plt.hist2d(emb[:, 0], emb[:, 1], bins=256, norm=LogNorm());


# It seems like here is small number of original variables, that combines into this dense rectangles. Wide empty lines can be interpreted like suffisient difference in some of original variable values, maybe categorical features.

# ## Scatter plot colored by target

# In[ ]:


plt.figure(figsize=(10, 10))
plt.scatter(emb[:, 0], emb[:, 1], marker='.', c=train['target'].values, alpha=0.1);


# Hmmm...
