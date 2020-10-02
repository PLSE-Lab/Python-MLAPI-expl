#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train=pd.read_csv("../input/nlp-getting-started/train.csv")
test=pd.read_csv("../input/nlp-getting-started/test.csv")


# In[ ]:


train.head()


# In[ ]:


train=train[['text','target']]


# In[ ]:


X=train.drop('target',axis=1)
y=train['target']


# In[ ]:


print(X.shape)
print(y.shape)


# In[ ]:


tf.__version__


# In[ ]:


from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from keras.layers import Dropout


# In[ ]:


vocab_size=5000


# In[ ]:


train.isnull().sum()


# In[ ]:



import nltk
import re
from nltk.corpus import stopwords


# In[ ]:


from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()
corpus = []
for i in range(0, len(train)):
    review = re.sub('[^a-zA-Z]', ' ', train['text'][i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
corpus[0]    


# In[ ]:


onehot_repr=[one_hot(words,voc_size)for words in corpus] 
onehot_repr[0]


# In[ ]:



sent_length=20
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs[0])


# In[ ]:


embedding_vector_features=40
from tensorflow.keras.layers import Dropout
## Creating model
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# somehow after using dropout accuracy is a bit lower ..

# In[ ]:


import numpy as np
X_final=np.array(embedded_docs)
y_final=np.array(y)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.30, random_state=42)


# In[ ]:


model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)


# In[ ]:


y_pred=model.predict_classes(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)*100


# will continue working on accuracy and make predictions for the test data ...

# In[ ]:


# test=test[['text']]
# test


# In[ ]:




