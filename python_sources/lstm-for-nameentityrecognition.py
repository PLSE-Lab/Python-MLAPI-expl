#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# 
# **Importing the dataset for named entity recognition model**

# In[ ]:


dframe = pd.read_csv("../input/ner.csv", encoding = "ISO-8859-1", error_bad_lines=False)


# In[ ]:


dframe


# ## Data preprocessing**

# In[ ]:


dframe.columns


# * Keep word, pos, sentence_idx and tag as an input, remove all other columns

# In[ ]:


dataset=dframe.drop(['Unnamed: 0', 'lemma', 'next-lemma', 'next-next-lemma', 'next-next-pos',
       'next-next-shape', 'next-next-word', 'next-pos', 'next-shape',
       'next-word', 'prev-iob', 'prev-lemma', 'prev-pos',
       'prev-prev-iob', 'prev-prev-lemma', 'prev-prev-pos', 'prev-prev-shape',
       'prev-prev-word', 'prev-shape', 'prev-word',"pos"],axis=1)


# In[ ]:


dataset.info()


# In[ ]:


dataset.head()


# In[ ]:


dataset.isnull().sum()


# Dropping rows having NULL Values

# In[ ]:


dataset.dropna(inplace=True)


# In[ ]:


dataset['tag'].value_counts()


# In[ ]:


dataset['tag'].describe()


# In[ ]:


dataset['tag'].unique()


# In[ ]:


dataset=dataset.drop(['shape'],axis=1)


# In[ ]:


dataset.head(10)


# > **Create list of list of tuples to differentiate each sentence from each other**

# In[ ]:


class SentenceGetter(object):
    
    def __init__(self, dataset):
        self.n_sent = 1
        self.dataset = dataset
        self.empty = False
        agg_func = lambda s: [(w, t) for w,t in zip(s["word"].values.tolist(),s["tag"].values.tolist())]
        self.grouped = self.dataset.groupby("sentence_idx").apply(agg_func)
        self.sentences = [s for s in self.grouped]


# In[ ]:


getter = SentenceGetter(dataset)


# In[ ]:


sentences = getter.sentences


# In[ ]:


print(sentences[5])


# In[ ]:


maxlen = max([len(s) for s in sentences])
print ('Maximum sequence length:', maxlen)


# In[ ]:


# Check how long sentences are so that we can pad them
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("ggplot")


# In[ ]:


plt.hist([len(s) for s in sentences], bins=40)
plt.show()


# In[ ]:


words = list(set(dataset["word"].values))
words.append("ENDPAD")


# In[ ]:


n_words = len(words); n_words


# In[ ]:


tags = list(set(dataset["tag"].values))


# In[ ]:


n_tags = len(tags)
n_tags


# **Converting words to numbers and numbers to words**

# In[ ]:


word2idx = {w: i for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}


# In[ ]:


word2idx['Obama']


# In[ ]:


tag2idx["O"]


# In[ ]:


from keras.preprocessing.sequence import pad_sequences
X = [[word2idx[w[0]] for w in s] for s in sentences]


# In[ ]:


X = pad_sequences(maxlen=140, sequences=X, padding="post",value=n_words - 1)


# In[ ]:


X[0]


# In[ ]:


y = [[tag2idx[w[1]] for w in s] for s in sentences]


# In[ ]:


y = pad_sequences(maxlen=140, sequences=y, padding="post", value=tag2idx["O"])


# In[ ]:


from keras.utils import to_categorical
y = [to_categorical(i, num_classes=n_tags) for i in y]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[ ]:


from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional


# In[ ]:


input = Input(shape=(140,))
model = Embedding(input_dim=n_words, output_dim=140, input_length=140)(input)
model = Dropout(0.2)(model)
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
model = Dropout(0.2)(model)
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
model = Dropout(0.5)(model)
out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)  # softmax output layer


# In[ ]:


model = Model(input, out)


# In[ ]:


model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


model.summary()


# In[ ]:


history = model.fit(X_train, np.array(y_train), batch_size=32, epochs=1, validation_split=0.2, verbose=1)


# In[ ]:


i = 0
p = model.predict(np.array([X_test[i]]))
p = np.argmax(p, axis=-1)
print("{:14} ({:5}): ({:4})".format("Word", "True", "Pred"))
for w,pred in zip(X_test[i],p[0]):
    print("{:14}: {}".format(words[w],tags[pred]))


# In[ ]:


p[0]


# In[ ]:


predictions = model.predict(np.array(X_test))
predictions = np.argmax(predictions, axis=-1)


# In[ ]:


predictions.shape


# In[ ]:


y_test_arr = np.array(y_test)
y_test_result=np.argmax(y_test_arr, axis=-1)
y_test_result.shape


# In[ ]:


y_test_result[0]


# In[ ]:


predictions[0]


# In[ ]:




