#!/usr/bin/env python
# coding: utf-8

# ## This model has the following characteristics:
# * No feature engineering
# * Extracting features from raw transactions through applying one-hot-encoding
# * Modeling a simple neural network without hidden layers

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import keras


# ### Read Data

# In[ ]:


df_train = pd.read_csv('../input/X_train.csv', encoding='cp949')
df_test = pd.read_csv('../input/X_test.csv', encoding='cp949')
y_train = pd.read_csv('../input/y_train.csv').gender
IDtest = df_test.custid.unique()

df_train.head()


# ### Transform Data with One-hot Encoding

# In[ ]:


level = 'brd_nm'

df_all = pd.concat([df_train, df_test])
X_train = pd.pivot_table(df_all, index='custid', columns=level, values='tot_amt',
                         aggfunc=lambda x: np.where(len(x) >=1, 1, 0), fill_value=0). \
                         reset_index(). \
                         query('custid not in @IDtest'). \
                         drop(columns=['custid']).values
X_test = pd.pivot_table(df_all, index='custid', columns=level, values='tot_amt',
                         aggfunc=lambda x: np.where(len(x) >=1, 1, 0), fill_value=0). \
                         reset_index(). \
                         query('custid in @IDtest'). \
                         drop(columns=['custid']).values

max_features = X_train.shape[1]


# ### Build Models

# In[ ]:


from keras import models
from keras import layers
from keras.optimizers import RMSprop
from keras import regularizers
from keras.callbacks import EarlyStopping

model = models.Sequential()
model.add(layers.Dense(1, input_shape=(max_features,), kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Activation('sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=100, batch_size=64, 
                    validation_split=0.2, callbacks=[EarlyStopping(patience=5)])

plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="validation loss")
plt.legend()
plt.title("Loss")
plt.show()


# ### Make Submissions

# In[ ]:


pred = model.predict(X_test)[:,0]
fname = 'submissions.csv'
submissions = pd.concat([pd.Series(IDtest, name="custid"), pd.Series(pred, name="gender")] ,axis=1)
submissions.to_csv(fname, index=False)
print("'{}' is ready to submit." .format(fname))


# ## End
