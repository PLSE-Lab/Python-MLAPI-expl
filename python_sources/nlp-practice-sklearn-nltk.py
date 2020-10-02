#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import re
import string

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train=pd.read_csv('../input/nlp-getting-started/train.csv')
test=pd.read_csv('../input/nlp-getting-started/test.csv')
sample_submission=pd.read_csv('../input/nlp-getting-started/sample_submission.csv')


# In[ ]:


train.head()


# In[ ]:


sns.countplot(train['target'])


# In[ ]:


print('train shape : ', train.shape)
print('test shape : ', test.shape)


# In[ ]:


print(train.isnull().sum(), '\n', '-'*40)
print(test.isnull().sum())


# * train['keywords'] values
# * train['location'] values
# 
# * when taget==1 and target==0

# In[ ]:


target_1_keyword=train.loc[train['target']==1]['keyword'].value_counts()[:20]
target_0_keyword=train.loc[train['target']==0]['keyword'].value_counts()[:20]

f, axes=plt.subplots(2,1, figsize=(25,10))
ax=axes.ravel()
sns.barplot(x=target_1_keyword.index, y=target_1_keyword.values, ax=ax[0])
sns.barplot(x=target_0_keyword.index, y=target_0_keyword.values, ax=ax[1])


# In[ ]:


target_1_location=train.loc[train['target']==1]['location'].value_counts()[:20]
target_0_location=train.loc[train['target']==0]['location'].value_counts()[:20]

f, axes=plt.subplots(2,1, figsize=(25,10))
ax=axes.ravel()
sns.barplot(x=target_1_location.index, y=target_1_location.values, ax=ax[0])
sns.barplot(x=target_0_location.index, y=target_0_location.values, ax=ax[1])


# * I think location column is not usefull. so drop! 

# In[ ]:


train=train.drop(columns='location')
test=test.drop(columns='location')


# * fill the Nan value of keyword feature to unknown

# In[ ]:


train['keyword']=train['keyword'].fillna('unknown')
test['keyword']=test['keyword'].fillna('unknown')


# In[ ]:


train.head()


# ## At NLP, process this works
# * lowercase translation
# * remove number
# * remove punctuation
# * remove stopword
# * remove special character
# * normalization(stemming or lemmatization)

# In[ ]:


#first, lowercaste translation, remove number, remove punctutation. then tokenize

def lower(text):#lowercase translation
    return text.lower()

def remove_number(text):#remove number
    new_text=re.sub(r'[0-9]+','',text)
    return new_text

def remove_punctuation(text):#remove punctutation
    table=str.maketrans('', '', string.punctuation)
    return text.translate(table)

train['text']=train['text'].apply(lambda x:lower(x))
test['text']=test['text'].apply(lambda x:lower(x))

train['text']=train['text'].apply(lambda x:remove_number(x))
test['text']=test['text'].apply(lambda x:remove_number(x))

train['text']=train['text'].apply(lambda x:remove_punctuation(x))
test['text']=test['text'].apply(lambda x:remove_punctuation(x))

#tokenzie
from nltk.tokenize import word_tokenize

train['text']=train['text'].apply(lambda x:word_tokenize(x))
test['text']=test['text'].apply(lambda x:word_tokenize(x))


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


# After tokenize, remove stopword, remove special text, normalization(stemming or lemmatization)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

def remove_stopword(text):
    new_text=[i for i in text if i not in stopwords.words('english')]
    return new_text

def trans_Lemmatize(text):
    return [WordNetLemmatizer().lemmatize(i) for i in text]

def trans_stem(text):
    return [PorterStemmer().stem(i) for i in text]

# In normalization, i just do lemmatize
train['text']=train['text'].apply(lambda i:remove_stopword(i))
test['text']=test['text'].apply(lambda i:remove_stopword(i))

train['text']=train['text'].apply(lambda i:trans_Lemmatize(i))
test['text']=test['text'].apply(lambda i:trans_Lemmatize(i))


# In[ ]:


# for CountVectorize, join the list
train['text']=train['text'].apply(lambda i:' '.join(i))
test['text']=test['text'].apply(lambda i:' '.join(i))


# In[ ]:


train.head()


# ## keyword -- OneHotEncoding
# ## text -- CountVectorizing

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer

ct=ColumnTransformer([('onehotencoder', OneHotEncoder(), ['keyword']),
                     ('dropout', 'drop', ['id','text'])])

X_train_onehot=ct.fit_transform(train[['id','keyword','text']])
X_test_onehot=ct.transform(test[['id','keyword','text']])

cv=CountVectorizer(min_df=5)

X_train_cv=cv.fit_transform(train['text'])
X_test_cv=cv.transform(test['text'])


# In[ ]:


from scipy import sparse

X_train=sparse.hstack((X_train_onehot, X_train_cv))
X_test=sparse.hstack((X_test_onehot, X_test_cv))

y_train=train['target']


# In[ ]:


X_train


# In[ ]:


X_test


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

forest=RandomForestClassifier()
forest.fit(X_train, y_train)


# In[ ]:


sub=pd.DataFrame({'id':test['id'],
                 'target':forest.predict(X_test)})

sub.to_csv('submission.csv', index=False)


# In[ ]:




