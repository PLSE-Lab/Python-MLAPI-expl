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

import os
print(os.listdir("../input"))
import numpy as np 
import pandas as pd 
import nltk
import os
import gc
from keras.preprocessing import sequence,text
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense,Dropout,Embedding,LSTM,Conv1D,GlobalMaxPooling1D,Flatten,MaxPooling1D,GRU,SpatialDropout1D,Bidirectional
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
#pd.set_option('display.max_colwidth',100)
pd.set_option('display.max_colwidth', -1)

# Any results you write to the current directory are saved as output.


# In[ ]:


train=pd.read_csv('../input/train.tsv',sep='\t')
train.head()


# In[ ]:


print(train.shape)


# In[ ]:


test=pd.read_csv('../input/test.tsv',sep='\t')

test.head()


# In[ ]:


print(test.shape)


# In[ ]:


sub=pd.read_csv('../input/sampleSubmission.csv')
sub.head()


# In[ ]:


test['Sentiment']=-999
test.head()


# In[ ]:


df=pd.concat([train,test],ignore_index=True)
print(df.shape)
df.tail()


# In[ ]:


del train,test
gc.collect()


# In[ ]:


from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.stem import SnowballStemmer,WordNetLemmatizer
stemmer=SnowballStemmer('english')
lemma=WordNetLemmatizer()
from string import punctuation
import re


# In[ ]:


def clean_review(review_col):
    review_corpus=[]
    for i in range(0,len(review_col)):
        review=str(review_col[i])
        review=re.sub('[^a-zA-Z0-9]',' ',review)
        #review=[stemmer.stem(w) for w in word_tokenize(str(review).lower())]
        review=[lemma.lemmatize(w) for w in word_tokenize(str(review).lower())]
        review=' '.join(review)
        review_corpus.append(review)
    return review_corpus


# In[ ]:


df['clean_review']=clean_review(df.Phrase.values)
df.head()


# In[ ]:


df_train=df[df.Sentiment!=-999]
df_train.shape


# In[ ]:


df_train.head()


# In[ ]:


df_test=df[df.Sentiment==-999]
df_test.drop('Sentiment',axis=1,inplace=True)
print(df_test.shape)
df_test.head()


# In[ ]:


del df
gc.collect()


# In[ ]:


train_text=df_train.clean_review.values
test_text=df_test.clean_review.values
target=df_train.Sentiment.values
y=to_categorical(target)
print(train_text.shape,target.shape,y.shape)


# In[ ]:


X_train_text,X_val_text,y_train,y_val=train_test_split(train_text,y,test_size=0.2,stratify=y,random_state=123)
print(X_train_text.shape,y_train.shape)
print(X_val_text.shape,y_val.shape)


# ###  Finding number of unique words in train set
# 
# 
# 
# 

# In[ ]:


all_words=' '.join(X_train_text)
all_words=word_tokenize(all_words)
dist=FreqDist(all_words)
num_unique_word=len(dist)
num_unique_word


# ### Finding max length of a review in train set

# In[ ]:


r_len=[]
for text in X_train_text:
    word=word_tokenize(text)
    l=len(word)
    r_len.append(l)
    
MAX_REVIEW_LEN=np.max(r_len)
MAX_REVIEW_LEN


# ###  Building Keras LSTM model
# 
# 
# 
# 

# In[ ]:


max_features = num_unique_word
max_words = MAX_REVIEW_LEN
batch_size = 128
epochs = 5
num_classes=5


# ### Tokenize Text

# In[ ]:


tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train_text))
X_train = tokenizer.texts_to_sequences(X_train_text)
X_val = tokenizer.texts_to_sequences(X_val_text)
X_test = tokenizer.texts_to_sequences(test_text)


# ### sequence padding

# In[ ]:


X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_val = sequence.pad_sequences(X_val, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
print(X_train.shape,X_val.shape,X_test.shape)


# ### 1. LSTM model

# In[ ]:


model1=Sequential()
model1.add(Embedding(max_features,100,mask_zero=True))
model1.add(LSTM(64,dropout=0.4, recurrent_dropout=0.4,return_sequences=True))
model1.add(LSTM(32,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))
model1.add(Dense(num_classes,activation='softmax'))
model1.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
model1.summary()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history1=model1.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=epochs, batch_size=batch_size, verbose=1)')


# In[ ]:


y_pred1=model1.predict_classes(X_test,verbose=1)


# In[ ]:


### plot the accuray
def train_validation(x, vy, ty, ax, colors=['b']):
    ax.plot(x, vy, 'b', label="Validation Loss")
    ax.plot(x, ty, 'r', label="Train Loss")
    plt.legend()
    plt.grid()
    fig.canvas.draw()
    
fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; 
ax.set_ylabel('Binary Crossentropy Loss')
x = list(range(1,epochs+1))
vy = history1.history['val_loss']
ty = history1.history['loss']
train_validation(x, vy, ty, ax)


# In[ ]:


sub.Sentiment=y_pred1
sub.to_csv('sub1.csv',index=False)
sub.head()


# ### 2. CNN

# In[ ]:


model2= Sequential()
model2.add(Embedding(max_features,100,input_length=max_words))
model2.add(Dropout(0.2))

model2.add(Conv1D(64,kernel_size=3,padding='same',activation='relu',strides=1))
model2.add(GlobalMaxPooling1D())

model2.add(Dense(128,activation='relu'))
model2.add(Dropout(0.2))

model2.add(Dense(num_classes,activation='softmax'))


model2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model2.summary()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history2=model2.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=epochs, batch_size=batch_size, verbose=1)')


# In[ ]:


### plot the accuray
def train_validation(x, vy, ty, ax, colors=['b']):
    ax.plot(x, vy, 'b', label="Validation Loss")
    ax.plot(x, ty, 'r', label="Train Loss")
    plt.legend()
    plt.grid()
    fig.canvas.draw()
    
fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; 
ax.set_ylabel('Binary Crossentropy Loss')
x = list(range(1,epochs+1))
vy = history2.history['val_loss']
ty = history2.history['loss']
train_validation(x, vy, ty, ax)


# In[ ]:


y_pred2=model2.predict_classes(X_test, verbose=1)


# In[ ]:


sub.Sentiment=y_pred2
sub.to_csv('sub2_cnn.csv',index=False)
sub.head()


# ### 3. CNN +GRU

# In[ ]:


model3= Sequential()
model3.add(Embedding(max_features,100,input_length=max_words))
model3.add(Conv1D(64,kernel_size=3,padding='same',activation='relu'))
model3.add(MaxPooling1D(pool_size=2))
model3.add(Dropout(0.25))
model3.add(GRU(128,return_sequences=True))
model3.add(Dropout(0.3))
model3.add(Flatten())
model3.add(Dense(128,activation='relu'))
model3.add(Dropout(0.5))
model3.add(Dense(5,activation='softmax'))
model3.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
model3.summary()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history3=model3.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=epochs, batch_size=batch_size, verbose=1)')


# In[ ]:


### plot the accuray
def train_validation(x, vy, ty, ax, colors=['b']):
    ax.plot(x, vy, 'b', label="Validation Loss")
    ax.plot(x, ty, 'r', label="Train Loss")
    plt.legend()
    plt.grid()
    fig.canvas.draw()
    
fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; 
ax.set_ylabel('Binary Crossentropy Loss')
x = list(range(1,epochs+1))
vy = history3.history['val_loss']
ty = history3.history['loss']
train_validation(x, vy, ty, ax)


# In[ ]:


y_pred3=model3.predict_classes(X_test, verbose=1)


# In[ ]:


sub.Sentiment=y_pred3
sub.to_csv('sub3_cnn+gru.csv',index=False)
sub.head()


# ### 4. Bidirectional GRU

# In[ ]:


model4 = Sequential()

model4.add(Embedding(max_features, 100, input_length=max_words))
model4.add(SpatialDropout1D(0.25))
model4.add(Bidirectional(GRU(128)))
model4.add(Dropout(0.5))

model4.add(Dense(5, activation='softmax'))
model4.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model4.summary()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history4=model4.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=epochs, batch_size=batch_size, verbose=1)')


# In[ ]:


### plot the accuray
def train_validation(x, vy, ty, ax, colors=['b']):
    ax.plot(x, vy, 'b', label="Validation Loss")
    ax.plot(x, ty, 'r', label="Train Loss")
    plt.legend()
    plt.grid()
    fig.canvas.draw()
    
fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; 
ax.set_ylabel('Binary Crossentropy Loss')
x = list(range(1,epochs+1))
vy = history4.history['val_loss']
ty = history4.history['loss']
train_validation(x, vy, ty, ax)


# In[ ]:


y_pred4=model4.predict_classes(X_test, verbose=1)


# In[ ]:


sub.Sentiment=y_pred4
sub.to_csv('sub4_bidirectional+gru.csv',index=False)
sub.head()


# In[ ]:




