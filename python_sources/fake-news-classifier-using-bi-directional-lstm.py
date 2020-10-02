#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.preprocessing.text import one_hot
from keras.layers import LSTM,Dense,Bidirectional

import nltk
import re
from nltk.corpus import stopwords


# In[ ]:


train = pd.read_csv('../input/fake-news/train.csv')
test = pd.read_csv('../input/fake-news/test.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.isna().sum()


# In[ ]:


train = train.dropna()


# In[ ]:


train.isna().sum()


# In[ ]:


test.isna().sum()


# In[ ]:


test = test.dropna()


# In[ ]:


test.isna().sum()


# In[ ]:


X = train.drop('label',axis=1)
X


# In[ ]:


y = train['label']
y


# In[ ]:


plt.style.use('fivethirtyeight')
sns.countplot(data=train,x='label')


# In[ ]:


X.shape


# In[ ]:


voc_size = 5000


# In[ ]:


messages = X.copy()


# In[ ]:


messages['title'][1]


# In[ ]:


messages.reset_index(inplace=True)


# In[ ]:


nltk.download('stopwords')


# In[ ]:


from nltk.stem import WordNetLemmatizer


# In[ ]:


lm = WordNetLemmatizer()


# In[ ]:


corpus = []

for i in range(0,len(messages)):
    review = re.sub('^[a-zA-Z]',' ',messages['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [lm.lemmatize(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[ ]:


test_data = test.copy()


# In[ ]:


test_data.reset_index(inplace=True)


# In[ ]:


test_data['title'][1]


# In[ ]:





# In[ ]:


test_corpus = []

for i in range(0,len(test_data)):
    review = re.sub('^[a-zA-Z]',' ',test_data['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [lm.lemmatize(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    test_corpus.append(review)


# In[ ]:


test_corpus


# In[ ]:


corpus


# In[ ]:


onehot_test_rep = [one_hot(words,voc_size) for words in test_corpus]
onehot_test_rep


# In[ ]:


onehot_rep = [one_hot(words,voc_size) for words in corpus]
onehot_rep


# In[ ]:


sent_length = 25
embedded_test_docs = pad_sequences(onehot_test_rep,padding='pre',maxlen=sent_length)
print(embedded_test_docs)


# In[ ]:


sent_length = 25
embedded_docs = pad_sequences(onehot_rep,padding='pre',maxlen=sent_length)
print(embedded_docs)


# In[ ]:


embedded_docs[0]


# In[ ]:


embedded_vector_features = 40
model = Sequential()
model.add(Embedding(voc_size,embedded_vector_features,input_length=sent_length))
model.add(Bidirectional(LSTM(100)))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


# In[ ]:


len(embedded_docs),y.shape


# In[ ]:


X_test_final = np.array(embedded_test_docs)


# In[ ]:


X_final = np.array(embedded_docs)
y_final = np.array(y)


# In[ ]:


X_final.shape,y_final.shape


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X_final,y_final,test_size=0.3,random_state=42)


# In[ ]:


model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=64,epochs=20)


# In[ ]:


y_pred = model.predict_classes(X_test_final)
y_pred


# In[ ]:


y_pred = np.array(y_pred)
y_pred


# In[ ]:




