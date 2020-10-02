#!/usr/bin/env python
# coding: utf-8

# In[114]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[115]:


from keras import backend as K
K.tensorflow_backend._get_available_gpus()


# In[116]:


df = pd.read_csv('../input/imdb-review-dataset/imdb_master.csv',encoding="latin-1")
df.head()


# In[117]:


df = df.drop(['Unnamed: 0','file'],axis=1)
df.columns = ['type',"review","sentiment"]
df.head()


# In[118]:


df = df[df.sentiment != 'unsup']
df['sentiment'] = df['sentiment'].map({'pos': 1, 'neg': 0})
df.head()


# In[119]:


import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

#stop_words = set(stopwords.words("english")) 
#lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text

#df['Processed_Reviews'] = df.review.apply(lambda x: clean_text(x))


# In[120]:


#df.head()


# In[121]:


#df.Processed_Reviews.apply(lambda x: len(x.split(" "))).mean()


# In[131]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D, SpatialDropout1D
from keras import initializers, regularizers, constraints, optimizers, layers
from keras import callbacks


# In[123]:


df_train = df[df.type == 'train']
df_test = df[df.type == 'test']


# In[124]:


# max_features = 6000
# tokenizer = Tokenizer(num_words=max_features)
# tokenizer.fit_on_texts(df['Processed_Reviews'])
# list_tokenized_train = tokenizer.texts_to_sequences(df_train['Processed_Reviews'])
# list_tokenized_test = tokenizer.texts_to_sequences(df_test['Processed_Reviews'])


# In[125]:


max_features = 6000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(df['review'])
list_tokenized_train = tokenizer.texts_to_sequences(df_train['review'])
list_tokenized_test = tokenizer.texts_to_sequences(df_test['review'])


# In[126]:


maxlen = 130
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
y = df_train['sentiment']


# In[139]:


embed_size = 128
model = Sequential()
model.add(Embedding(max_features, embed_size))
model.add(SpatialDropout1D(0.4))
model.add(Bidirectional(LSTM(100, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(100, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[140]:


batch_size = 128
epochs = 10
cb = []
cb.append(callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=0, mode='auto', baseline=None, restore_best_weights=True))
model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks = cb)


# In[141]:


y_test = df_test['sentiment']
y_test.head()


# In[142]:


X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
prediction = model.predict(X_te)
y_pred = (prediction > 0.5)


# In[138]:


from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_pred, y_test)
print('Accuracy ',accuracy_score(y_test, y_pred))
pd.DataFrame(cm)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


# In[ ]:


def ngram_vectorize(train_texts, train_labels, val_texts):
    kwargs = {
        'ngram_range' : (1, 2),
        'dtype' : 'int32',
        'strip_accents' : 'unicode',
        'decode_error' : 'replace',
        'analyzer' : 'word',
        'min_df' : 2,
    }
    
    tfidf_vectorizer = TfidfVectorizer(**kwargs)
    x_train = tfidf_vectorizer.fit_transform(train_texts)
    x_val = tfidf_vectorizer.transform(val_texts)
    
    selector = SelectKBest(f_classif, k=min(6000, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train).astype('float32')
    x_val = selector.transform(x_val).astype('float32')
    return x_train, x_val


# In[ ]:


df_bag_train, df_bag_test = ngram_vectorize(df_test['review'], df_test['sentiment'], df_train['review'])


# In[ ]:


nb = MultinomialNB()
nb.fit(df_bag_train, y)
nb_pred = nb.predict(df_bag_test)
print(classification_report(y_test, nb_pred))
cm = confusion_matrix(nb_pred, y_test)
print('Accuracy ',accuracy_score(y_test, nb_pred))
pd.DataFrame(cm)

