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


# ## Reading the dataset

# In[ ]:


train = pd.read_csv("../input/train_E6oV3lV.csv")
test = pd.read_csv("../input/test_tweets_anuFYb8.csv")


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train['label'] = train['label'].astype('category')


# In[ ]:


train.info()


# ## Processing the Tweets
# 
# - Remove the special characters, numbers etc. (keep only alphabets)
# - lemmatize the text
# 

# In[ ]:


from nltk.stem import WordNetLemmatizer
from nltk import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import re


# In[ ]:


train['text_lem'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]',' ',text)) for text in lis]) for lis in train['tweet']]
test['text_lem'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]',' ',text)) for text in lis]) for lis in test['tweet']]


# In[ ]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(train['text_lem'],train['label'])


# In[ ]:


vect = TfidfVectorizer(ngram_range = (1,4)).fit(X_train)


# In[ ]:


vect_transformed_X_train = vect.transform(X_train)
vect_transformed_X_test = vect.transform(X_test)


# In[ ]:


from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score


# ### F1 score is used as an evaluation measure as, when the data is skewed like in this case, where the number of hate speech tweets are very less, accuracy cannot be relied upon.

# In[ ]:


modelSVC = SVC(C=100).fit(vect_transformed_X_train,y_train)


# In[ ]:


predictionsSVC = modelSVC.predict(vect_transformed_X_test)
sum(predictionsSVC==1),len(y_test),f1_score(y_test,predictionsSVC)


# In[ ]:


modelLR = LogisticRegression(C=100).fit(vect_transformed_X_train,y_train)


# In[ ]:


predictionsLR = modelLR.predict(vect_transformed_X_test)
sum(predictionsLR==1),len(y_test),f1_score(y_test,predictionsLR)


# In[ ]:


modelNB = MultinomialNB(alpha=1.7).fit(vect_transformed_X_train,y_train)


# In[ ]:


predictionsNB = modelNB.predict(vect_transformed_X_test)
sum(predictionsNB==1),len(y_test),f1_score(y_test,predictionsNB)


# In[ ]:


modelRF = RandomForestClassifier(n_estimators=20).fit(vect_transformed_X_train,y_train)


# In[ ]:


predictionsRF = modelRF.predict(vect_transformed_X_test)
sum(predictionsRF==1),len(y_test),f1_score(y_test,predictionsRF)


# In[ ]:


modelSGD = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3).fit(vect_transformed_X_train,y_train)


# In[ ]:


predictionsSGD = modelSGD.predict(vect_transformed_X_test)
sum(predictionsSGD==1),len(y_test),f1_score(y_test,predictionsSGD)


# Based on all the above models trained we conclude that the logistic regression(C=100) is the better model amoung them, ergo, we use this model as our final model.

# In[ ]:


vect = TfidfVectorizer(ngram_range = (1,4)).fit(train['text_lem'])
vect_transformed_train = vect.transform(train['text_lem'])
vect_transformed_test = vect.transform(test['text_lem'])


# In[ ]:


FinalModel = LogisticRegression(C=100).fit(vect_transformed_train,train['label'])


# In[ ]:


predictions = FinalModel.predict(vect_transformed_test)


# In[ ]:


submission = pd.DataFrame({'id':test['id'],'label':predictions})


# In[ ]:


file_name = 'test_predictions.csv'
submission.to_csv(file_name,index=False)


# In[ ]:




