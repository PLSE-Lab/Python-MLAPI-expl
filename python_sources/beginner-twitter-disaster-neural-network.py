#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
df.columns


# In[ ]:


X = df['text']
y = df['target']


# In[ ]:


import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk import word_tokenize,sent_tokenize
lem = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# In[ ]:


def text_preprocess(text):
#     clean_text = text.lower()
    clean_text = re.sub('[^A-Za-z ]+', "", text).lower()
    clean_text = re.sub(r'(http|https|pic.)\S+', " ", clean_text)
    clean_words = word_tokenize(clean_text)
    words = set([lem.lemmatize(word) for word in clean_words if not word in stop_words])
    final_text = ' '.join(words)
    return final_text
    


# In[ ]:


text = "This is a sample sentence, # showing off the ? not stop words stop filtration. # https://www.nnjkj.com"
sentence = text_preprocess(text)
sentence


# In[ ]:


X = X.apply(text_preprocess)


# In[ ]:


X.shape


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer
from sklearn.model_selection import train_test_split
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(X)


# In[ ]:


X.shape


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 0)


# In[ ]:


X_train.shape


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout


# In[ ]:


model = Sequential()
model.add(Dense(1024, activation='relu', input_dim=(5000)))
model.add(Dropout(0.9))
model.add(Dense(1024,activation = 'relu'))
model.add(Dropout(0.9))
model.add(Dense(512,activation = 'relu'))
model.add(Dropout(0.9))
model.add(Dense(1, activation='sigmoid'))


# In[ ]:


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[ ]:


model.fit(X_train, y_train,
          batch_size=15,
          epochs=60,
          validation_data=(X_test, y_test))


# In[ ]:


df_test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
test = df_test['text']


# In[ ]:


test = test.apply(text_preprocess)


# In[ ]:


test = tfidf.transform(test)


# In[ ]:


pred = model.predict_classes(test)
target = pred.reshape(3263)


# In[ ]:


submission = pd.DataFrame({
    'id' : df_test["id"].astype('int64'),
    'target' : target
})

# submission.to_csv('SVMC_55th.csv', index=False)
submission.to_csv('target.csv', index=False)


# In[ ]:




