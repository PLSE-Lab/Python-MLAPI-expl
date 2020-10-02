#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import os

import keras
from keras.models import Sequential, Input, Model
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout, TimeDistributed, Flatten
from keras.preprocessing.sequence import pad_sequences

print(os.listdir("../input"))


# In[ ]:


dframe = pd.read_csv("../input/ner.csv", encoding = "ISO-8859-1", error_bad_lines=False)
dframe.head()


# In[ ]:


dataset=dframe.drop(['Unnamed: 0', 'lemma', 'next-lemma', 'next-next-lemma', 'next-next-pos',
       'next-next-shape', 'next-next-word', 'next-pos', 'next-shape',
       'next-word', 'prev-iob', 'prev-lemma', 'prev-pos',
       'prev-prev-iob', 'prev-prev-lemma', 'prev-prev-pos', 'prev-prev-shape',
       'prev-prev-word', 'prev-shape', 'prev-word','shape'],axis=1)


# In[ ]:


dataset.head(10)


# In[ ]:


class SentenceGetter(object):
    
    def __init__(self, dataset):
        self.n_sent = 1
        self.dataset = dataset
        self.empty = False
        agg_func = lambda s: [(w, t) for w,t in zip(s["word"].values.tolist(), s["tag"].values.tolist())]
        self.grouped = self.dataset.groupby("sentence_idx").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


# In[ ]:


getter = SentenceGetter(dataset)
sentences = getter.sentences
sentences[1]


# In[ ]:


sentences = [[s[0].lower() for s in sent] for sent in getter.sentences]
sentences[1]


# In[ ]:


labels = [[s[1] for s in sent] for sent in getter.sentences]
print(labels[1])


# In[ ]:


maxlen = max([len(s) for s in sentences])
print ('Maximum sequence length:', maxlen)


# In[ ]:


# Check how long sentences are so that we can pad them
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("ggplot")


# In[ ]:


plt.hist([len(s) for s in sentences], bins=50)
plt.show()


# In[ ]:


words = np.array([x.lower() if isinstance(x, str) else x for x in dataset["word"].values])
words = list(set(words))
words.append('unk')
words.append('pad')
n_words = len(words); n_words


# In[ ]:


tags = list(set(dataset["tag"].values))
n_tags = len(tags)
n_tags


# In[ ]:


word2idx = {w: i for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}


# In[ ]:


tag2idx


# In[ ]:


word2idx['unk']


# In[ ]:


X = [[word2idx.get(w,'27420') for w in s] for s in sentences]
X[1]


# In[ ]:


y = [[tag2idx.get(l) for l in lab] for lab in labels]
y[1]


# In[ ]:


X = pad_sequences(maxlen=140, sequences=X, padding="post", value=n_words-1)


# In[ ]:


word2idx['pad'], n_words-1


# In[ ]:


y = pad_sequences(maxlen=140, sequences=y, padding="post", value=tag2idx["O"])


# In[ ]:


y[1]


# In[ ]:


X[1]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# In[ ]:


y_train = keras.utils.to_categorical(y_train)
print(X_train.shape, y_train.shape)


# In[ ]:


model = Sequential()

model.add(Embedding(n_words, 50))
model.add(Bidirectional(LSTM(140, return_sequences=True)))
model.add(Bidirectional(LSTM(140, return_sequences=True)))
model.add(TimeDistributed(Dense(n_tags, activation="softmax")))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, batch_size=150, epochs=5, verbose=1, validation_split=0.2)


# In[ ]:


pred = model.predict(X_test)   


# In[ ]:


pred


# **Functional API**

# In[ ]:


# input = Input(shape=(140,))
# model = Embedding(n_words, 50)(input)
# model = Bidirectional(LSTM(units=140, return_sequences=True))(model)
# model = Bidirectional(LSTM(units=140, return_sequences=True))(model)
# # reshape = keras.layers.Reshape((-1, 140, 1))(model)

# out = TimeDistributed(Dense(18, activation="softmax"))(model)  # softmax output layer
# model = Model(input, out)

# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# model.fit(X_train, y_train, batch_size=150, epochs=1, verbose=1)


# In[ ]:


model.summary()


# In[ ]:


i = 5
p = model.predict(np.array([X_test[i]]))
p = np.argmax(p, axis=-1)
print("{:14} ({:5}): {}".format("Word", "True", "Pred"))
for w,pred in zip(X_test[i],p[0]):
    print("{:14}: {}".format(words[w],tags[pred]))


# In[ ]:




