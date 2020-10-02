#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (6,6)

# from keras import backend as K
# from keras.engine.topology import Layer
# from keras import initializers, regularizers, constraints

# from keras.layers import Dense, Input, LSTM, Bidirectional, Activation, Conv1D, GRU, TimeDistributed
# from keras.layers import Dropout, Embedding, GlobalMaxPooling1D, MaxPooling1D, Add, Flatten, SpatialDropout1D
# from keras.layers import GlobalAveragePooling1D, BatchNormalization, concatenate
# from keras.layers import Reshape, merge, Concatenate, Lambda, Average
# from keras.models import Sequential, Model, load_model
# from keras.callbacks import ModelCheckpoint
# from keras.initializers import Constant
# from keras.layers.merge import add

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import os
print(os.listdir("../input"))


# In[ ]:


# load data

# df = pd.read_json('../input/News_Category_Dataset_v2.json')
df = pd.read_json('../input/News_Category_Dataset_v2.json', lines=True)
print(df.shape)
df.head()


# # prepare data

# In[ ]:


cates = df.groupby('category')
print("total categories:", cates.ngroups)
print(cates.size())


# In[ ]:


# as shown above, THE WORLDPOST and WORLDPOST should be the same category, so merge them.

df.category = df.category.map(lambda x: "WORLDPOST" if x == "THE WORLDPOST" else x)


# In[ ]:


# using headlines and short_description as input X

df['text'] = df.headline + " " + df.short_description

# tokenizing

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df.text)
X = tokenizer.texts_to_sequences(df.text)
df['words'] = X

# delete some empty and short data

df['word_length'] = df.words.apply(lambda i: len(i))
df = df[df.word_length >= 6]

df.head()


# In[ ]:


print(df.shape)


# In[ ]:


df.word_length.describe()


# In[ ]:


df['category'].value_counts()


# In[ ]:


df = df.drop_duplicates(subset="text")
print(df.shape)


# In[ ]:


df.columns


# In[ ]:


df.head()


# In[ ]:


df.drop(["words"],axis=1).to_csv("Huffpost_News_Category.csv.gz",index=False,compression="gzip")


# In[ ]:




