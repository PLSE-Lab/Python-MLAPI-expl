#!/usr/bin/env python
# coding: utf-8

# In[171]:


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
get_ipython().run_line_magic('matplotlib', 'inline')

models = keras.models
layers = keras.layers


# In[172]:


data = pd.read_csv('../input/bbc-text.csv')


# In[173]:


data.head()


# In[174]:


data['category'].value_counts()


# In[175]:


max_words=1000
tokenize = keras.preprocessing.text.Tokenizer(num_words=max_words, char_level=False)


# In[176]:


x_train, x_test, y_train, y_test = train_test_split(
    data['text'],
    data['category'],
    test_size=0.2,
    random_state=42
)


# In[177]:


tokenize.fit_on_texts(x_train)
x_train = tokenize.texts_to_matrix(x_train)
x_test = tokenize.texts_to_matrix(x_test)


# In[178]:


encoder = LabelEncoder()
encoder.fit(data['category'])
y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)


# In[179]:


num_classes = int(np.max(y_train) + 1)


# In[180]:


y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[181]:


batch_size = 32
epochs = 2
drop_ratio = 0.5


# In[182]:


[max_words, num_classes]


# In[183]:


model = models.Sequential()
model.add(layers.Dense(512, input_shape=(max_words,)))
model.add(layers.Activation('relu'))
model.add(layers.Dense(num_classes))
model.add(layers.Activation('softmax'))


# In[184]:


model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# In[185]:


x_train.shape


# In[186]:


history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_split=0.1
)


# In[187]:


score = model.evaluate(
    x_test,
    y_test,
    batch_size=batch_size,
    verbose=1
)


# In[188]:


print('Test loss: ', score[0])
print('Test accuracy: ', score[1])

