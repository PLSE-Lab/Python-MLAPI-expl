#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


train_df=pd.read_csv('../input/train.tsv',sep='\t')
test_df=pd.read_csv('../input/test.tsv',sep='\t')


# In[3]:


train_df.head()


# In[4]:


from nltk.tokenize.treebank import TreebankWordTokenizer


# In[5]:


tbwt=TreebankWordTokenizer()
def tokenizer(row):
    ltokens=tbwt.tokenize(row['Phrase'])
    #print(row['Phrase'])
    tok_sen=' '.join(ltokens)
    #print(ltokens)
    return tok_sen


# In[6]:


train_df['token']=train_df.apply(tokenizer,axis=1)
test_df['token']=test_df.apply(tokenizer,axis=1)


# In[7]:


train_df.head()


# In[8]:


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def lemmatize(row):
    #print(row['token'])
    split=row['token'].split(' ')
    lem_sen=' '.join([lemmatizer.lemmatize(w) for w in split])
    #print(lem_sen)
    return lem_sen


# In[9]:


train_df['lemma']=train_df.apply(lemmatize,axis=1)
test_df['lemma']=test_df.apply(lemmatize,axis=1)


# In[10]:


train_df.head()


# In[11]:


from keras.preprocessing.text import Tokenizer
token=Tokenizer()
token.fit_on_texts(train_df['lemma'])


# In[12]:


X_train=token.texts_to_sequences(train_df['lemma'])
X_train


# In[13]:


def mlen(row):
    s=row['lemma'].split(' ')
    return len(s)

train_df['len']=train_df.apply(mlen,axis=1)
train_df.head()


# In[14]:


max(train_df['len'])


# In[15]:


test_df['len']=test_df.apply(mlen,axis=1)
max(test_df['len'])


# In[16]:


from keras.preprocessing.sequence import pad_sequences
X_train=pad_sequences(X_train,maxlen=56,padding='post')


# In[17]:


from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
Y_train=train_df.iloc[:,3]
Y_train=Y_train.as_matrix()
Y_train=Y_train.reshape(-1,1)
Y_tr=ohe.fit_transform(Y_train)


# In[18]:


from keras.models import Sequential
from keras.layers import Dense,Dropout,Embedding,LSTM


# In[19]:


len(token.word_index)


# In[21]:


model=Sequential()
model.add(Embedding(14219,300))
model.add(LSTM(150))
model.add(Dense(5,activation='softmax'))


# In[22]:


model.summary()


# In[25]:


model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[26]:


model.fit(X_train,Y_tr,batch_size=64,epochs=15)


# In[30]:


X_test=token.texts_to_sequences(test_df['lemma'])
X_test=pad_sequences(X_test,maxlen=56,padding='post')


# In[31]:


X_test.shape


# In[32]:


s=model.predict(X_test)
s=np.argmax(s,axis=1)
print(s)
s=pd.DataFrame(s)
s['PhraseId']=test_df['PhraseId']
s.columns=['Sentiment','PhraseId']
s=s[['PhraseId','Sentiment']]
s.to_csv('submissions.csv',index=False)


# In[33]:


s.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




