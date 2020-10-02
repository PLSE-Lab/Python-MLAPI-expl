#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


file_path = '../input/IMDB Dataset.csv'
data = pd.read_csv(file_path)
data.head()


# In[3]:


y = data.sentiment
y.head()


# In[4]:


label = {'positive':1, 'negative':-1}

def preprocess_y(sentiment):
    return label[sentiment]

y = y.apply(preprocess_y)
y.head()


# In[5]:


X = data.review
X.head()


# In[6]:


import nltk
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))


# In[7]:


import re
def preprocess(review):
    #convert the tweet to lower case
    review.lower()
    #convert all urls to sting "URL"
    review = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',review)
    #convert all @username to "AT_USER"
    review = re.sub('@[^\s]+','AT_USER', review)
    #correct all multiple white spaces to a single white space
    review = re.sub('[\s]+', ' ', review)
    #convert "#topic" to just "topic"
    review = re.sub(r'#([^\s]+)', r'\1', review)
    tokens = word_tokenize(review)
    tokens = [w for w in tokens if not w in stop_words]
    return " ".join(tokens)

X = X.apply(preprocess)
X.head()


# In[8]:


from sklearn.feature_extraction.text import TfidfVectorizer
def feature_extraction(data):
    tfv=TfidfVectorizer(sublinear_tf=True, stop_words = "english")
    features=tfv.fit_transform(data)
    pickle.dump(tfv.vocabulary_, open("svm_feature.pkl", "wb"))
    return features

data = np.array(X)
label = np.array(y)
features = feature_extraction(data)

print(features)


# In[9]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size = 0.20) 


# In[10]:


from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear')  

svclassifier.fit(X_train, y_train)  


# In[11]:


from sklearn.metrics import accuracy_score
val_pred = svclassifier.predict(X_test)
#print(val_pred)
print(accuracy_score(y_test, val_pred))


# In[12]:


filename = 'svm_model.sav'
pickle.dump(svclassifier, open(filename, 'wb'))


# Saving the model

# In[13]:


loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)


# Loading the saved model and testing

# In[14]:


text = 'you are really beautiful'
text = preprocess(text)
print(text)
text = np.array([text])
print(text)

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
tfv_loaded = TfidfVectorizer(sublinear_tf=True, stop_words = "english", vocabulary=pickle.load(open("svm_feature.pkl", "rb")))
text = transformer.fit_transform(tfv_loaded.fit_transform(text))
print(text)
polarity = loaded_model.predict(text)
print(polarity)

