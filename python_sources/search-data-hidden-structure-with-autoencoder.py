#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import StandardScaler

from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Dropout, Flatten, Input
from keras import backend as K
import keras
from matplotlib.colors import LogNorm


# In[ ]:


folder_path = '/kaggle/input/ieee-fraud-detection/'


# In[ ]:


train = pd.read_csv(f'{folder_path}train_transaction.csv')
test = pd.read_csv(f'{folder_path}test_transaction.csv')


# # Prepare data and model
# I use only numeric cols. In order to use categorical features we need to get their embeddings first. 
# Time and ID is not included, because I want to check is there information about time in other features.

# In[ ]:


cats = ['ProductCD',
    'card1',
    'card2',
    'card3',
    'card4',
    'card5',
    'card6',
    'P_emaildomain',
    'R_emaildomain',
    'M1',
    'M2',
    'M3',
    'M4',
    'M5',
    'M6',
    'M7',
    'M8',
    'M9',
    'addr1',
    'addr2']

cols = list(train.columns)[3:]
nocats = [c for c in cols if (not c in cats)]


# In[ ]:


get_ipython().run_cell_magic('time', '', 'ss = StandardScaler(copy=False)\ndata_ss = ss.fit_transform(np.nan_to_num(train[nocats].values))')


# ### Model
# Tanh in the output of the decoder is good choice for visualisation: all objects will be projected on square from -1 to 1.

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


data_ss = np.clip(data_ss, -10, 10)


# In[ ]:


get_ipython().run_cell_magic('time', '', "epochs = 50\nbatch_size = 9548\nhistory = c_autoencoder.fit(data_ss, data_ss,\n                    epochs=epochs,\n                    batch_size=batch_size,\n                    shuffle=True,\n                    verbose=1)\n\nloss_history = history.history['loss']\nplt.figure(figsize=(10, 5))\nplt.plot(loss_history);")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'emb = c_encoder.predict(data_ss)')


# # Let's look at 2d density
# It seems like there is some clusters

# In[ ]:


plt.figure(figsize=(10, 10))
plt.hist2d(emb[:, 0], emb[:, 1], bins=256, norm=LogNorm());


# ### How fraudent transactions is distributed?
# Looks like fraud is distributed almost like normal transactions, but there is some regions where fraudent transactions appear more often.

# In[ ]:


plt.figure(figsize=(20, 20))
plt.scatter(emb[:, 0], emb[:, 1], c=train['isFraud'].values,
           marker='.', alpha=0.1);


# ### Is there time leak in numerical features?
# Obviously there is information about time.

# In[ ]:


plt.figure(figsize=(20, 20))
plt.scatter(emb[:, 0], emb[:, 1], c=train['TransactionDT'].values,
           marker='.', alpha=0.1, cmap='jet');


# ### What causes clustering?
# Colouring by ProductCD gives answer for this question. Remember that categorical features was not included in visualization. It means that numerical features distribution in different categories is different. I think, we need to separate models for different ProductCD.

# In[ ]:


prd_d = {p: i for i, p in enumerate(train['ProductCD'].unique())}


# In[ ]:


plt.figure(figsize=(20, 20))
plt.scatter(emb[:, 0], emb[:, 1],
            c=train['ProductCD'].apply(lambda x: prd_d[x]).values,
           marker='.', alpha=0.1, cmap='jet');


# # What about test?
# All the same.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'test_ss = ss.transform(np.nan_to_num(test[nocats].values))')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'test_emb = c_encoder.predict(test_ss)')


# In[ ]:


plt.figure(figsize=(20, 20))
plt.scatter(test_emb[:, 0], test_emb[:, 1],
            c=test['TransactionDT'].values,
           marker='.', alpha=0.1, cmap='jet');


# # THAT'S ALL, FOLKS!
