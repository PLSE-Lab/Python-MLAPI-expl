#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import keras


# In[ ]:


df = pd.read_json('../input/train.json')


# In[ ]:


def normalise_and_pad(sequence):
    ret = np.pad(np.array(sequence) / 255, ((0, 10-len(sequence)),(0,0)), 'edge')
    return ret


# In[ ]:


x_train = np.asarray([normalise_and_pad(x) for x in df['audio_embedding']])
y_train = df['is_turkey'].values


# In[ ]:


input_layer = keras.layers.Input(shape=(10,128))
gru_out = keras.layers.GRU(128)(input_layer)
dense_out = keras.layers.Dense(1, activation='sigmoid')(gru_out)


# In[ ]:


model = keras.models.Model(input_layer, dense_out)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, validation_split=0.2, epochs=100, batch_size=256)


# In[ ]:


df_test = pd.read_json('../input/test.json')


# In[ ]:


x_test = np.asarray([normalise_and_pad(x) for x in df_test['audio_embedding']])
y_test = model.predict(x_test)


# In[ ]:


df_out = pd.DataFrame({'vid_id':df_test['vid_id'],'is_turkey':[x[0] for x in y_test]})


# In[ ]:


df_out.to_csv('submission.csv', index=False)

