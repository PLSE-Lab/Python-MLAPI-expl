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

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/train.tsv',delimiter='\t')
train.head()


# In[ ]:


test = pd.read_csv('../input/test.tsv',delimiter='\t')
test.head()


# In[ ]:


train.shape


# In[ ]:


train.dtypes


# In[ ]:


train.describe()


# In[ ]:


train.info()


# In[ ]:


train.columns


# In[ ]:


train.Sentiment.nunique()


# In[ ]:


train.Sentiment.unique()


# In[ ]:


train.isnull().sum()


# In[ ]:


train.PhraseId.duplicated().sum()


# In[ ]:


train.PhraseId.nunique()


# In[ ]:


train.SentenceId.nunique()


# In[ ]:


train.SentenceId.duplicated().sum()


# In[ ]:


train.SentenceId.value_counts()[:10]


# In[ ]:


train[train.SentenceId==1][:5]


# In[ ]:


train[train.SentenceId==128][:5]


# In[ ]:


train.Sentiment.value_counts()


# most of the movies reviewd as snetiment value 2 and 3 next to that.

# In[ ]:


train.Phrase[0]


# In[ ]:


train[['Sentiment','Phrase']].sort_values('Sentiment',ascending=True)[:10]


# we will try to convert text to int form nd before that Phrase is in in Object type we have to change that first

# In[ ]:


train["Phrase"] = train["Phrase"].astype(str,copy=True) 


# In[ ]:


type(train.Phrase)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion


# In[ ]:


cv = CountVectorizer()


# In[ ]:


x = train.Phrase
x


# In[ ]:


y = train.Sentiment.values
y


# In[ ]:



x_cv  = cv.fit_transform(x)
x_cv


# In[ ]:


type(x_cv)


# In[ ]:


x_cv.toarray()


# In[ ]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# In[ ]:


lr.fit(x_cv,y)


# In[ ]:


lr.score(x_cv,y)


# In[ ]:


x_test = test.Phrase
x_test


# In[ ]:


x_test_cv = cv.transform(x_test)
x_test_cv


# In[ ]:


x_test_cv.toarray()


# In[ ]:


preds = lr.predict(x_test_cv)
preds


# In[ ]:


type(preds)


# In[ ]:


lr.score(x_cv,y)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB,GaussianNB
mnb = MultinomialNB()
gnb = GaussianNB()


# In[ ]:


mnb.fit(x_cv,y)
mnb.score(x_cv,y)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,ExtraTreesClassifier,RandomForestClassifier


# In[ ]:


dtc = DecisionTreeClassifier()
etc = ExtraTreesClassifier()
abc = AdaBoostClassifier()
rfc = RandomForestClassifier()
c_list = [dtc,etc,abc,rfc]


# In[ ]:


scores = []
for i in c_list:
    i.fit(x_cv,y)
    score = i.score(x_cv,y)
    scores.append(score)
print(scores)


# In[ ]:


for i,j in zip(c_list,scores):
    print('{} score is {}'.format(str(i),j))      


# In[ ]:


print('the best classifier score  is {}'.format(max(scores)))


# Here the best classifier is extratrees clasifier, we achieved this score with default values and further can be improved with tuning and also it is based on countvec method

#  more in pipeline, next time we will see other embending methods and scores with diff algos

# **If you like it, please upvote for me.**
# 
# Thank you : )
