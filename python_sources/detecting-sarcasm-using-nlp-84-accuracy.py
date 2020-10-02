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


data=pd.read_json('//kaggle//input//news-headlines-dataset-for-sarcasm-detection//Sarcasm_Headlines_Dataset_v2.json',lines=True)
data.head()


# In[ ]:


df=pd.read_json('//kaggle//input//news-headlines-dataset-for-sarcasm-detection//Sarcasm_Headlines_Dataset.json',lines=True)
df.head()


# In[ ]:


import nltk
import re


# In[ ]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
ps = PorterStemmer()
lemmatizer=WordNetLemmatizer()


# In[ ]:


corpus = []
for i in range(0, len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['headline'][i])
    review = review.lower()
    #review = review.split()
    
    #review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
    #review = ' '.join(review)
    corpus.append(review)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,ngram_range=(1,3))
X = cv.fit_transform(corpus).toarray()


# In[ ]:


y=df['is_sarcastic']
y.head()


# # #using PAC with k-fold cross validation 

# In[ ]:



from sklearn.linear_model import PassiveAggressiveClassifier
linear_clf = PassiveAggressiveClassifier()


# In[ ]:


linear_clf.fit(X, y)


# In[ ]:


from sklearn.model_selection import cross_val_score
score_PAC = cross_val_score(linear_clf,X,y,cv=10)


# In[ ]:


score_PAC


# In[ ]:


score_PAC.mean()


# # using naive_bayes with train_test_split

# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB().fit(x_train,y_train)


# In[ ]:


y_pred=model.predict(x_test)


# In[ ]:


from sklearn import metrics


# In[ ]:


score=metrics.accuracy_score(y_test,y_pred)


# In[ ]:


score


# # using naive_bayes with k-fold cross validation

# In[ ]:


model1=MultinomialNB().fit(X,y)


# In[ ]:


score_naive = cross_val_score(model1,X,y,cv=10)


# In[ ]:


score_naive


# In[ ]:


score_naive.mean()


# # using PAC with train_test_split 

# In[ ]:


model2=PassiveAggressiveClassifier()


# In[ ]:


model2.fit(x_train,y_train)


# In[ ]:


pred=model2.predict(x_test)


# In[ ]:


score_=metrics.accuracy_score(y_test,pred)


# In[ ]:


score_


# # so, here we can observe that on this data naive_bayes with k-fold cross validation works for NLP, as it is getting higher accuracy.

# In[ ]:


s=1
for i in range(1,10001):
    s=i*s
print(s)


# In[ ]:


len(str(s))


# In[ ]:




