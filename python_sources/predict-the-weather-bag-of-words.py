#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
from nltk.corpus import stopwords
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


train.head()


# In[ ]:


refined_train = train
refined_test = test


# In[ ]:


refined_train["tweet"].head()


# In[ ]:


def refine_tweets(tweet):
    tweet = re.sub("@mention", "", tweet)
    tweet = re.sub("[^A-Za-z0-9]", " ", tweet)
    tweet = re.sub("  ", " ", tweet)
    print(tweet)
    return tweet


# In[ ]:


#refined_train["tweet"] = refine_tweets(refined_train["tweet"][0])
#refine_tweets(refined_train["tweet"][100])
refined_train["tweet"] = [refine_tweets(tweet) for tweet in refined_train["tweet"]]
refined_test["tweet"] = [refine_tweets(tweet) for tweet in refined_test["tweet"]]


# In[ ]:


refined_train.head()


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=100, strip_accents='unicode', analyzer='word', stop_words='english')


# In[ ]:


X = tfidf.fit_transform(refined_train["tweet"])
test_X = tfidf.fit_transform(refined_test["tweet"])


# In[ ]:


y = refined_train.iloc[:,4:]


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets, linear_model, multioutput
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb


# In[ ]:


xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)


# In[ ]:


forest = RandomForestClassifier(n_estimators = 100) 
#mor = linear_model.LinearRegression()
mor = multioutput.MultiOutputRegressor(GradientBoostingRegressor(n_estimators=50))
#mor = multioutput.MultiOutputRegressor(AdaBoostRegressor(n_estimators=50))
#mor = multioutput.MultiOutputRegressor(xg_reg)
#mor = multioutput.MultiOutputRegressor(xgb.XGBRegressor(n_estimators = 50, max_depth=10))


# In[ ]:


mor.fit( X_train, y_train )


# In[ ]:


predictions = mor.predict(X_test)


# In[ ]:


print("Score:", mor.score(X_test, y_test))


# In[ ]:


mor.fit( X, y )


# In[ ]:


predictions_test = mor.predict(test_X)


# In[ ]:


predictions_test


# In[ ]:


variableNames = ['s1','s2','s3','s4','s5','w1','w2','w3','w4','k1','k2','k3','k4','k5','k6','k7','k8','k9','k10','k11','k12','k13','k14','k15']


# In[ ]:


predictionsFrame = pd.DataFrame(data=predictions_test,columns=variableNames)


# In[ ]:


submissionFrame = pd.concat([test['id'], predictionsFrame], axis=1)


# In[ ]:


submissionFrame


# In[ ]:


submissionFrame.to_csv( "Weather.csv", index=False)


# In[ ]:




