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


# In[ ]:


data=pd.read_csv("../input/sms-spam-collection-dataset/spam.csv", encoding='latin-1')


# In[ ]:


data.head()


# In[ ]:


data=data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1) #droping unnamed columns
data.head()


# In[ ]:


data['label']=data['v1'].map({'ham':0, 'spam':1}) #create new column named label where spam is 1 and ham is 0
data.head()


# In[ ]:


#calculating number of spam and ham sms
no_of_spam=data['v1'].value_counts()
no_of_spam  


# In[ ]:


percentage_spam=(no_of_spam[1]/(no_of_spam[1]+no_of_spam[0]))
print("percentage of spam sms: ", percentage_spam*100, "%")


# In[ ]:


X=data['v2']
y=data['label']
print(X)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


# In[ ]:


#dividing dataset into training and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1)

#creating vec, an object of CountVectorizer and removing the stop words
#then printing the vocabulary obtained after removing stop words
vec=CountVectorizer(stop_words='english')
vec.fit(X_train)
vec.vocabulary_


# In[ ]:


X_train_transformed=vec.transform(X_train)
X_test_transform=vec.transform(X_test)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import metrics


# In[ ]:


#first making model using multinomial naive bayes
mnb=MultinomialNB()
mnb.fit(X_train_transformed, y_train)

#getting predictions
pred_mnb=mnb.predict(X_test_transform)

#getting probability for the binary classification of sms
prob_mnb=mnb.predict_proba(X_test_transform)


# In[ ]:


#accuracy score for multinomial naive bayes model
metrics.accuracy_score(y_test, pred_mnb)


# In[ ]:


#creating confusion matrix for multinomial naive bayes model
# [[True Negative  ,  False Positive]
#  [False Negative ,  True Positive]]
confusion=metrics.confusion_matrix(y_test, pred_mnb)
print(confusion)


# In[ ]:


# printing precision, recall score and f1 score for the confusion matrix
# obtained by multinomial naive bayes model
print("Precision: ", metrics.precision_score(y_test, pred_mnb))
print("Recall Score: ", metrics.recall_score(y_test, pred_mnb))
print("F1 Score: ", metrics.f1_score(y_test, pred_mnb))


# In[ ]:


# From the confusion matrix we are getting 9 false positive, ie, 9 ham sms would be classified as spam which is not a very good metric


# In[ ]:


# now creating another model using bernoulli naive bayes
bnb=BernoulliNB()
bnb.fit(X_train_transformed, y_train)

# getting predictions
pred_bnb=bnb.predict(X_test_transform)

#getting probability for the binary classification of sms
prob_bnb=bnb.predict_proba(X_test_transform)


# In[ ]:


# printing accuracy score of bernoulli naive bayes model
metrics.accuracy_score(y_test, pred_bnb)


# In[ ]:


# accuracy score of BernoulliNB model is slightly less than that of MultinomialNB


# In[ ]:


#creating confusion matrix for multinomial naive bayes model
# [[True Negative  ,  False Positive]
#  [False Negative ,  True Positive]]
confusion_bnb=metrics.confusion_matrix(y_test, pred_bnb)
print(confusion_bnb)


# In[ ]:


# printing precision, recall score and f1 score of bernoulli naive bayes model
print("Precision: ", metrics.precision_score(y_test, pred_bnb))
print("Recall Score: ", metrics.recall_score(y_test, pred_bnb))
print("F1 Score: ", metrics.f1_score(y_test, pred_bnb))


# In[ ]:


# from the confusion matrix we are getting 0 false positive result


# In[ ]:




