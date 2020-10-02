#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df_yelp=pd.read_csv('/kaggle/input/sentiment-labelled-sentences-data-set/yelp_labelled.txt',sep='\t',header=None)


# In[ ]:


df_yelp.head()


# In[ ]:


df_imdb=pd.read_csv('/kaggle/input/sentiment-labelled-sentences-data-set/imdb_labelled.txt',sep='\t',header=None)


# In[ ]:


df_amzn=pd.read_csv('/kaggle/input/sentiment-labelled-sentences-data-set/amazon_cells_labelled.txt',sep='\t',header=None)


# In[ ]:


df_yelp.shape, df_amzn.shape , df_imdb.shape


# In[ ]:


col_names=['review','sentiment']
df_yelp.columns=col_names
df_imdb.columns=col_names
df_amzn.columns=col_names


# In[ ]:


df_yelp.head()


# In[ ]:


df_yelp.loc[20]['review']


# In[ ]:


data=df_yelp.append([df_amzn,df_imdb],ignore_index=True)


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data['sentiment'].value_counts()


# In[ ]:


data.isnull().sum()


# # now data cleaning

# In[ ]:


import string
punc=string.punctuation


# In[ ]:


punc


# In[ ]:


import spacy
nlp=spacy.load('en_core_web_sm')


# ### lemmatization 

# In[ ]:


x='hello! as its WorlD'
doc=nlp(x)
for token in doc:
    print(token.lemma_)


# In[ ]:


def lemmatize(x):
    doc=nlp(x)
    tokens=[]
    for token in doc:
        if token.lemma_ != '-PRON-':
            temp=token.lemma_.lower().strip()
        else:
            temp=token.lower_
        tokens.append(temp)
    return tokens


# In[ ]:


lemmatize(x)


# ### removing stop words ans punctuations

# In[ ]:


from spacy.lang.en.stop_words import STOP_WORDS


# In[ ]:


print(STOP_WORDS)


# In[ ]:


import string
punc=string.punctuation


# In[ ]:


def stop_word_and_punc(x):
    tokens=[]
    for token in x:
        if token not in STOP_WORDS and token not in punc:
            tokens.append(token)
    return tokens


# In[ ]:


def data_cleaning(x):
    tokens=lemmatize(x)
    return stop_word_and_punc(tokens)


# In[ ]:


x="Hello my name is shubham and this is good learning drinking runs learned"
data_cleaning(x)


# ## entity visualization

# In[ ]:


data.head()


# In[ ]:


text=" ".join(data['review'])


# In[ ]:


text


# In[ ]:


from spacy import displacy
nlp=spacy.load('en_core_web_sm')


# In[ ]:


doc=nlp(text)


# In[ ]:


displacy.render(doc,style='pos')


# In[ ]:


displacy.render(doc,style='ent')


# ## analysing sentiment of word count and char count

# In[ ]:


import matplotlib.pyplot as plt


# ## word count

# In[ ]:


data['word_count']=data['review'].apply(lambda x:len(x.split()))


# In[ ]:


data.head()


# ### char count

# In[ ]:


def get_char_count(x):
    count=0
    for word in x.split():
        count+=len(word)
    return count


# In[ ]:


data['char_count']=data['review'].apply(lambda x:get_char_count(x))


# In[ ]:


data.head()


# ## plotting sentiment with word count and char count

# In[ ]:


plt.hist(data[data['sentiment']==0]['word_count'],bins=200)
plt.hist(data[data['sentiment']==1]['word_count'],bins=200)
plt.xlim([0,60])
plt.show()


# In[ ]:


plt.hist(data[data['sentiment']==0]['char_count'],bins=200)
plt.hist(data[data['sentiment']==1]['char_count'],bins=200)
plt.xlim([0,300])
plt.show()


# In[ ]:





# # vectorization with tfidf

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC,SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.pipeline import Pipeline


# In[ ]:


tfidf=TfidfVectorizer(tokenizer=data_cleaning)
classifier=SVC()


# In[ ]:


X=data['review']
y=data['sentiment']


# In[ ]:


X_train,X_test,y_train,y_test= train_test_split(X,y,shuffle=True,random_state=0,test_size=0.2)


# In[ ]:


X_train.shape, X_test.shape 


# In[ ]:


clf=Pipeline([('tfidf',tfidf),('clf',classifier)])


# In[ ]:


clf.fit(X_train,y_train)


# In[ ]:


y_pred=clf.predict(X_test)
report=classification_report(y_test,y_pred)
print(report)


# In[ ]:


cm=confusion_matrix(y_test,y_pred)
print(cm)


# # using neural network

# In[ ]:


import spacy
nlp=spacy.load('en_core_web_sm')


# In[ ]:


x="hello world apple mango"


# In[ ]:


doc=nlp(x)


# In[ ]:


for token in doc:
    print(token.text, token.has_vector , token.vector.shape)


# # now we will add vectors for all reviews

# In[ ]:


def get_vector(x):
    doc=nlp(x)
    return doc.vector.reshape(-1,1)


# In[ ]:


data['vector']=data['review'].apply(lambda x:get_vector(x))


# In[ ]:


data.head()


# In[ ]:


data.loc[3]['vector'].shape


# In[ ]:


import tensorflow as tf


# In[ ]:


X=np.concatenate(data['vector'].to_numpy(),axis=1)
X=np.transpose(X)
y=(data['sentiment']>1).astype(int)


# In[ ]:


X.shape , y.shape


# In[ ]:


from tensorflow.keras.layers import Dense,Dropout,BatchNormalization,LSTM
from tensorflow.keras.models import Sequential


# In[ ]:


model=Sequential([
    Dense(128,activation='relu'),
    Dropout(0.25),
    BatchNormalization(),
    Dense(64,activation='relu'),
    Dropout(0.25),
    BatchNormalization(),
    Dense(2,activation='sigmoid')
])


# In[ ]:


import tensorflow as tf
y_oh=tf.keras.utils.to_categorical(y,num_classes=2)


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y_oh,random_state=2,test_size=0.2)


# In[ ]:


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
history=model.fit(X_train,y_train,epochs=10,batch_size=32,validation_data=[X_test,y_test])


# In[ ]:


from sklearn.metrics import confusion_matrix
y_pred=model.predict(X_test)
y_pred=np.argmax(y_pred,axis=1)
y_pred.shape , y_test.shape


# In[ ]:


cm=confusion_matrix(y_test,y_pred)
cm


# In[ ]:





# In[ ]:




