#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import zipfile
from zipfile import ZipFile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import os

import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
import os
print(os.listdir("../input"))


# In[2]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sub = pd.read_csv('../input/sample_submission.csv')


# In[61]:


train.shape,test.shape


# In[62]:


num_words = 20000
max_len = 220
emb_size = 256
num_model = 3


# In[63]:


tokenizer=Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(list(train['comment_text']) + list(test['comment_text']))
Y=np.where(train['target']>=0.5,1,0)


# In[64]:


print(tokenizer.document_count)


# In[65]:


X = tokenizer.texts_to_sequences(list(train['comment_text']))
test_X = tokenizer.texts_to_sequences(list(test['comment_text']))


# In[66]:


X = sequence.pad_sequences(X, maxlen = max_len)
test_X = sequence.pad_sequences(test_X, maxlen = max_len)


# In[79]:


# create the model
embedding_vecor_length = emb_size
model = Sequential()
model.add(Embedding(num_words, embedding_vecor_length, input_length=X.shape[1]))
model.add(Dropout(0.25))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# In[80]:


score=[]
predictions=[]
for i in range(num_model):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20)
    es = EarlyStopping(monitor='val_loss', patience=100)
    model.fit(X_train,Y_train, validation_data=(X_test, Y_test), epochs=3, batch_size=1024,callbacks=[es])
    pred = model.predict(test_X)
    scores = model.evaluate(test_X, pred, verbose=0)
    score.append(scores)
    predictions.append(pred)
    print(i)


# In[81]:


score=pd.DataFrame(score)
score[0].mean()


# In[82]:


fina_pred=np.average(predictions,axis=0)
fina_pred=pd.DataFrame(fina_pred)


# In[83]:


ids=test['id']
ids=pd.DataFrame(ids)


# In[84]:


submission=pd.concat([ids,fina_pred],axis=1)
submission['prediction']=submission.iloc[:,1:]
submission=submission.drop([0],axis=1)
submission.to_csv("submission.csv",index=False)

