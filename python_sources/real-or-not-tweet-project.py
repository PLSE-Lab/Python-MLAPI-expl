#!/usr/bin/env python
# coding: utf-8

# This is kaggle real or not problem.I solve this NLP problem in a certains steps:
# 1.Importing data and Libaries
# 2.Removing stopwords and punctuations i.e Data preprocessing
# 3.Data Cleaning (Lemmatization)
# 4.Model Buiding
# 5.Training the model 
# 6.Hyperparater tuning
# 7.Final predictions
# 8.Submissions

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


# In[ ]:


#Importing data and libaries


# In[ ]:


import pandas as pd
sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
test = pd.read_csv("../input/nlp-getting-started/test.csv")
train = pd.read_csv("../input/nlp-getting-started/train.csv")


# In[ ]:


import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np


# In[ ]:



lemma=WordNetLemmatizer()

train.drop(['keyword','location'],inplace=True,axis=1)
test.drop(['keyword','location'],inplace=True,axis=1)


# In[ ]:


#data preprocessing
train_corpus=[]
test_corpus=[]

for  i in range(len(train)):
    review=re.sub('[^a-zA-Z]',' ',train['text'][i])
    review=review.lower()
    review=review.split()
    review=[lemma.lemmatize(word) for word in review if word not in stopwords.words('english')]
    review=' '.join(review)
    train_corpus.append(review)
    
for  j in range(len(test)):
    review1=re.sub('[^a-zA-Z]',' ',test['text'][j])
    review1=review1.lower()
    review1=review1.split()
    review1=[lemma.lemmatize(word) for word in review1 if word not in stopwords.words('english')]
    review1=' '.join(review1)
    test_corpus.append(review1)  


# In[ ]:


#Model Building
from sklearn.feature_extraction.text import TfidfVectorizer
 
tf=TfidfVectorizer(max_features=10000)


# In[ ]:


Xtrain2=tf.fit_transform(train_corpus).toarray()
 
Ytrain2=train['target']


# In[ ]:


Xtest2=tf.fit_transform(test_corpus).toarray()


# In[ ]:


Xtrain2


# In[ ]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(Xtrain2,Ytrain2,test_size=0.3,random_state=2)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
 
model=MultinomialNB()


# In[ ]:


model.fit(xtrain,ytrain)
predict1=model.predict(xtest)
print(model.score(xtest,ytest))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=150,random_state=2)
rfc.fit(xtrain,ytrain)
print(rfc.score(xtest,ytest))


# In[ ]:


#Hyperparameter tuning
from sklearn.model_selection import GridSearchCV
parameters={'n_estimators':[50,100,150,200,300]}
grid=GridSearchCV(rfc,parameters,cv=None)
grid.fit(xtrain,ytrain)


# In[ ]:


grid.best_params_
#Naive_bayes perform well so we select naive bayes model and make predictions


# In[ ]:


submit3=pd.DataFrame()
submit3['id']=test['id']   
submit3['target']=predictions_log2

submit3.to_csv('realornotsubmission555.csv',index=False)

