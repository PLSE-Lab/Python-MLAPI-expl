#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.random.seed(0)
plt.style.use("ggplot")

import tensorflow as tf
print('Tensorflow version:', tf.__version__)
print('GPU detected:', tf.config.list_physical_devices('GPU'))
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data = pd.read_csv('../input/ner-dataset/ner_datasetreference.csv', encoding='latin1')
data = data.fillna(method='ffill')
data.head(20)


# In[ ]:


data['Sentence #'].unique()


# In[ ]:


data.columns


# In[ ]:


print("unique words; ",data["Word"].nunique())
print("unique tags; ",data["Tag"].nunique())


# In[ ]:


#creating dataset for unique words, tags
words = list(set(data["Word"].values))
words.append("ENDPAD")
tags = list(set(data["Tag"].values))


# In[ ]:


np.shape(words)


# In[ ]:


num_words = len(words)
num_tags = len(tags)


# In[ ]:


num_words


# In[ ]:


class Sentence_getter(object):
    def __init__(self, data):
        self.data = data
        agg_fun = lambda s: [(w, p, t) for w,p,t in zip(s["Word"].values.tolist(),
                                                       s["POS"].values.tolist(),
                                                       s['Tag'].values.tolist())]
        
        self.grouped = self.data.groupby('Sentence #').apply(agg_fun)
        self.sentences = [i for i in self.grouped]
    
    


# In[ ]:


getter = Sentence_getter(data)


# In[ ]:


sentences = getter.sentences


# In[ ]:


sentences[0]


# In[ ]:


# Creating vocabulary
word2idx = {w: i+1 for i,w in enumerate(words)}
tag2idx = {t: i for i,t in enumerate(tags)}


# In[ ]:


tag2idx


# **Hist plot of len of sentences**

# In[ ]:


plt.hist([len(s) for s in sentences], bins = 50)
plt.plot()


# Lets take 50 as maxlen to be on safer side

# Padding inputs to maxlen
# 
# Creating INPUTS AND OUTPUTS

# In[ ]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

max_len = 50

X = [[word2idx[w[0]] for w in s] for s in sentences]
X = pad_sequences(maxlen = max_len, sequences=X, padding='post', value=num_words-1)

y = [[tag2idx[w[2]] for w in s] for s in sentences]
y = pad_sequences(maxlen = max_len, sequences=y, padding='post', value=tag2idx["O"])


# Converting output labes to categorical ( ONE_HOT )

# In[ ]:


y = [to_categorical(i, num_classes=num_tags) for i in y]


# In[ ]:


y[0]


# In[ ]:


x_train


# Now train-test splitting

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=1 )


# Building and compiling a BiLSTM

# In[ ]:


from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D, Bidirectional


# In[ ]:


input_word = Input(shape=(max_len,))
model = Embedding(input_dim=num_words, output_dim=max_len, input_length=max_len)(input_word)
model = SpatialDropout1D(0.1)(model)
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
out = TimeDistributed(Dense(num_tags, activation='softmax'))(model)

model = Model(input_word, out)
model.summary()


# In[ ]:


model.compile(optimizer='adam',loss='categorical_crossentropy',
             metrics=['accuracy'])


# TRAINING MODEL with per-epoch visualizations using CALLBACKS

# In[ ]:


pip install livelossplot


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping
from livelossplot import PlotLossesKeras


# In[ ]:


type(y_train)


# In[ ]:


early_stopping = EarlyStopping(monitor='val_accuracy', patience = 1, verbose=0, mode='max', restore_best_weights=False)
callbacks = [PlotLossesKeras(), early_stopping]

history = model.fit(
    x_train, np.array(y_train),
    validation_split=0.2,
    batch_size=32,
    epochs=3,
    verbose=1,
    callbacks=callbacks
)


# In[ ]:


model.evaluate(x_test, np.array(y_test))


# In[ ]:


x_test[0]


# In[ ]:


i = np.random.randint(0, x_test.shape[0])
p = model.predict(np.array([x_test[i]]))
# print(np.shape(p))
# print(p)
p = np.argmax(p, axis=-1)


y_true = np.argmax(np.array(y_test), axis=-1)[i]

print("{:15}{:5}\t{}\n".format("Word", "True", "Pred"))
print("-"*30)

for (w, t, pred) in zip(x_test[i], y_true, p[0]):
    print("{:15}{}\t{}".format(words[w-1], tags[t], tags[pred]))


# In[ ]:




