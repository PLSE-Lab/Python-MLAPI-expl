#!/usr/bin/env python
# coding: utf-8

# In this notebook we will be using a LSTM network to create a model to predict whether a news is fake or not.
# 
# I have also solved the same problem using Tf-Idf you can refer to this link and have a look.
# https://github.com/sid26ranjan/fake-news-classifier
# 
# Make sure you have enabled GPU in accelerator to speed up the training process and try various methods to achieve a different accuracy.
# 
# I have tried to explain the things that i have used in the notebook.feedbacks and suggestions are most welcomed.
# 
# Please upvote if you find this notebook useful.
# 
# 

# In[ ]:


import pandas as pd


# In[ ]:


data=pd.read_csv('../input/fake-news/train.csv')


# In[ ]:


data.head()

#we will be using the title column for our prediction


# In[ ]:


#checking for null values in the dataset

data.isnull().sum()


# In[ ]:


data.shape


# In[ ]:


#we will use the title column so other columns will be of no use

data=data.drop(['text','author','id'],axis=1)


# In[ ]:


#there are some  null values in the title column also

data.isnull().sum()


# In[ ]:


#as title is the only column is the what we are using if it contains NaN values we have to drop it.

data=data.dropna()


# In[ ]:


data.isnull().sum()


# In[ ]:


data.shape


# In[ ]:


data.head()


# In[ ]:


X=data['title']
y=data['label']


# In[ ]:


X.shape


# In[ ]:


#importing all necessary modules that we will be using to build our LSTM neural network

import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense


# In[ ]:





# In[ ]:


#we dropped some rows as there were nan values so reset index will make it uniform

X=X.reset_index()


# In[ ]:


X=X.drop(['index'],axis=1)


# In[ ]:


X.tail()


# In[ ]:


#as we dropped some rows so to make the dataframe in order
y=y.reset_index()


# In[ ]:


y=y.drop(['index'],axis=1)


# In[ ]:


y.tail()


# In[ ]:


# importing nltk,stopwords and porterstemmer we are using stemming on the text we have and stopwords will help in removing the stopwords in the text

#re is regular expressions used for identifying only words in the text and ignoring anything else
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[ ]:



ps = PorterStemmer()
corpus = []
#each row of the dataset is considered here.everything except the alphabets are removed ,stopwords are also being removed here .the text is converted in lowercase letters and stemming is performed
#lemmatisation can also be used here at the end a corpus of sentences is created
for i in range(0, len(X)):
    review = re.sub('[^a-zA-Z]', ' ',X['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[ ]:


corpus[30]


# In[ ]:


#vocabulary size
voc_size=5000


# In[ ]:


#performing onr hot representation

onehot_repr=[one_hot(words,voc_size)for words in corpus] 


# In[ ]:


len(onehot_repr[0])


# In[ ]:


len(onehot_repr[700])


# In[ ]:


#specifying a sentence length so that every sentence in the corpus will be of same length

sent_length=25

#using padding for creating equal length sentences


embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs)


# In[ ]:


#Creating model

from tensorflow.keras.layers import Dropout
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(Dropout(0.3))
model.add(LSTM(200))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[ ]:


import numpy as np
X_final=np.array(embedded_docs)
y_final=np.array(y)


# In[ ]:


X_final.shape,y_final.shape


# In[ ]:


#splitting the data for training and testing the model

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.10, random_state=42)


# In[ ]:


model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=20,batch_size=64)


# In[ ]:


#loading test dataset for prediction

test=pd.read_csv('../input/fake-news/test.csv')


# In[ ]:


test.head()


# In[ ]:


#null values in the test dataset

test.isnull().sum()


# In[ ]:


#using the title column only as we did in the train dataset

test=test.drop(['text','id','author'],axis=1)


# In[ ]:


test.head()


# In[ ]:


test.isnull().sum()


# In[ ]:


test.fillna('fake fake fake',inplace=True)

#the solution file that can be submitted in kaggle expects it to have 5200 rows so we can't drop rows in the test dataset


# In[ ]:


test.shape


# In[ ]:


#creating corpus for the test dataset exactly the same as we created for the training dataset

corpus_test = []
for i in range(0, len(test)):
    review = re.sub('[^a-zA-Z]', ' ',test['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus_test.append(review)


# In[ ]:


#creating one hot representation for the test corpus

onehot_repr_test=[one_hot(words,voc_size)for words in corpus_test] 


# In[ ]:


#padding for the test dataset
sent_length=25

embedded_docs_test=pad_sequences(onehot_repr_test,padding='pre',maxlen=sent_length)
print(embedded_docs_test)


# In[ ]:


X_test=np.array(embedded_docs_test)


# In[ ]:


#making predictions for the test dataset

check=model.predict_classes(X_test)


# In[ ]:


check


# In[ ]:


check.shape


# In[ ]:


test.shape


# In[ ]:


submit_sample=pd.read_csv('../input/fake-news/submit.csv')


# In[ ]:


submit_sample.head()


# In[ ]:


type(check)


# In[ ]:


check[0]


# In[ ]:


val=[]
for i in check:
    val.append(i[0])


# In[ ]:


#inserting our predicted values in the submission file

submit_sample['label']=val


# In[ ]:


submit_sample.head()


# In[ ]:


#saving the submission file

submit_sample.to_csv('submission.csv',index=False)


# if this notebook was helpful please upvote.

# In[ ]:




