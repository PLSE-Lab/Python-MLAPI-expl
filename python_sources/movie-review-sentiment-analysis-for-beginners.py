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


# Importing the train,test,submission datasets

# In[ ]:


test=pd.read_csv(r"/kaggle/input/movie-review-sentiment-analysis-kernels-only/test.tsv.zip",sep="\t")
train=pd.read_csv(r"/kaggle/input/movie-review-sentiment-analysis-kernels-only/train.tsv.zip",sep="\t")
sub = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv', sep=",")


# # **Data Analysis**

# In[ ]:


#lets look at the test data
test.head()


# In[ ]:


#lets look at the test data
train.head(100)


# ### **Output Inspection**
# * 0 - negative   
# * 1 - somewhat negative    
# * 2 - neutral   
# * 3 - somewhat positive   
# * 4 - positive   

# In[ ]:


# lets look at the shape of the train data
train.shape


# It is a huge dataset.

# In[ ]:


# lets look at the shape of the test data
test.shape


# In[ ]:


train.loc[train['SentenceId']==3]


# we can see that the sentenceId with value 3 has mostly same  repeated words in Phrase column 

# In[ ]:


train.loc[train['SentenceId']==2]


# Similarly,we can see that the sentenceId with value 2 has mostly same repeated words in Phrase column

# In[ ]:





# **Point to Note**
# 
# *       We can say here that each sentenceId is grouped  based on the similar words in phrase column

# In[ ]:


# since we have different values in  sentenceId,check the total no of unique sentenceId 
print("For train data ",train['SentenceId'].nunique()) 
print("For test data ",test['SentenceId'].nunique()) 


# In[ ]:



pd.DataFrame(train.groupby('SentenceId')['Phrase'].count()).head(10)


# In[ ]:


## Returning average count of phrases per sentence, per Dataset
int(train.groupby('SentenceId')['Phrase'].count().mean())


# In[ ]:


int(test.groupby('SentenceId')['Phrase'].count().mean())


# In[ ]:


#Returning average word length of phrases
print("train ",int(np.mean(train['Phrase'].apply(lambda x: len(x.split())))))
print("test",int(np.mean(test['Phrase'].apply(lambda x: len(x.split())))))


# # **Exploring Target Value**

# In[ ]:


train_count=train['Sentiment'].value_counts() 


# In[ ]:


#gets the unique value count of an object
train_labels=train['Sentiment'].value_counts().index


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
fig,ax = plt.subplots(1, 1, dpi = 100, figsize = (7, 5))
g=sns.barplot(train_labels,train_count)
ax.set_xlabel("target")
ax.set_ylabel("count")


# * we can say that almost half of the target values are 2(neutral).
# * we can also say that data is not balanced based on target feature

# # **Feature Engineering** 

# So, we have only phrases as data. And a phrase can contain a single word. And one punctuation mark can cause phrase to receive a different sentiment. Also assigned sentiments can be strange. This means several things:
# * using stopwords can be a bad idea, especially when phrases contain one single stopword
# * untuation could be important, so it should be used;
# * ngrams are necessary to get the most info from data (know about ngrams ->https://www.kaggle.com/c/avito-demand-prediction/discussion/58819)
# 

# In[ ]:


import nltk


# In[ ]:


tokenizer = nltk.tokenize.TweetTokenizer()


# **TF-IDF**

# In[ ]:


# import tfidf vectoriser
from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


vectorizer = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenizer.tokenize)
full_text = list(train['Phrase'].values) + list(test['Phrase'].values)
vectorizer.fit(full_text)  #learns both train and test data vocabulary


# **Why have i fitted the vectorizer with both train and test data?**
# > In real world we don't know what will be in new(test) data, so we have to fit only train data. On the other hand, in kaggle we have both train and test, this means we can    take   into account word distribution in both datasets.
# Also go through this link https://www.kaggle.com/questions-and-answers/58368 to know more

# In[ ]:


train_vectorized = vectorizer.transform(train['Phrase'])
test_vectorized = vectorizer.transform(test['Phrase'])


# In[ ]:


train_vectorized.shape


# In[ ]:


y = train['Sentiment']


# In[ ]:


test_vectorized.shape


# **Applying Model**

# In[ ]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression


# In[ ]:


logreg = LogisticRegression()
ovr = OneVsRestClassifier(logreg)


# Know about OneVsRestClassifier here https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html

# In[ ]:


ovr.fit(train_vectorized, y)


# In[ ]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(ovr, train_vectorized, y, scoring='accuracy', n_jobs=-1, cv=10)


# In[ ]:


print("Mean of 10 cv :",np.mean(scores) * 100)
print( "standard deviation",np.std(scores) * 100)


# In[ ]:


y_test=ovr.predict(test_vectorized)


# In[ ]:


sub.Sentiment=y_test


# In[ ]:


sub.head()


# In[ ]:


sub.to_csv('submission.csv',index=False)


# In[ ]:




