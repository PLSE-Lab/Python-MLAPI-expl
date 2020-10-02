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


df = pd.read_json('/kaggle/input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json',lines=True)


# In[ ]:


df.head()


# In[ ]:


print('Sarcastic: ',len(df[df['is_sarcastic'] == 1]))
print('Not Sarcastic: ',len(df[df['is_sarcastic'] == 1]))


# So, we have equally distributed data
# 
# deleting the article link as it is not important for our classification

# In[ ]:


del df['article_link']


# In[ ]:


df.head()


# In[ ]:


x = df['headline'].values
y = df['is_sarcastic'].values 


# In[ ]:


#removing punct
import string
punct = string.punctuation


# In[ ]:


cleaned_x = []
for word in x:
    sent = ''
    for char in word:
        if char not in punct:
            char = char.lower()
            sent += char
    cleaned_x.append(sent)


# ## Removing stop words.
# ### It was observed that by removing stop words the accuracy decreased. We were getting an accuracy of about 70% using bag of words model, and 71% using top 1000 n-grams

# In[ ]:


#later try removing stop words and check accuracy
import nltk
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))
cleaned_X = []
for sent in cleaned_x:
    sent_cleaned = ''
    for word in sent.split():
        if word not in stops:
            sent_cleaned += word
            sent_cleaned += ' '
    cleaned_X.append(sent_cleaned)


# In[ ]:





# # Trying different techniques
# 
# ## 1) Bag of words
# 
# - In this method we create a vocabulary using count vectorizer and select the top 1000 words for our classification

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


vectorizer = CountVectorizer(max_features=1000)
vectorizer.fit(cleaned_x)


# In[ ]:


x_data = vectorizer.transform(cleaned_x)


# In[ ]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x_data,y,random_state=1)
x_train.shape,y_train.shape,x_test.shape,y_test.shape


# ## Applying Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc = RandomForestClassifier(n_estimators=500,random_state=42)
rfc.fit(x_train,y_train)


# In[ ]:


preds = rfc.predict(x_test)


# In[ ]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[ ]:


print('accuracy: ', accuracy_score(y_test,preds))
print(confusion_matrix(y_test,preds))


# ## Trying N-Grams

# In[ ]:


count_vec = CountVectorizer(max_features = 1000, ngram_range=(1,2))
count_vec.fit(cleaned_x)


# In[ ]:


x_feat_ngrams = count_vec.transform(cleaned_x)


# In[ ]:


x_feat_ngrams = np.array(x_feat_ngrams.todense())


# In[ ]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x_feat_ngrams,y,random_state=1)
x_train.shape,y_train.shape,x_test.shape,y_test.shape


# In[ ]:


rfc = RandomForestClassifier(n_estimators=500,random_state=42)
rfc.fit(x_train,y_train)


# In[ ]:


preds = rfc.predict(x_test)


# In[ ]:


print('accuracy: ', accuracy_score(y_test,preds))
print(confusion_matrix(y_test,preds))


# ## Trying TF-IDF Model
# - This model tries to identify how important a word is in a corpus

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


tfIdfVec = TfidfVectorizer(max_features=1000)
tfIdfVec.fit(cleaned_x)


# In[ ]:


tf_idf_x = tfIdfVec.transform(cleaned_x)


# In[ ]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(tf_idf_x.todense(),y,random_state=1)
x_train.shape,y_train.shape,x_test.shape,y_test.shape


# In[ ]:


rfc = RandomForestClassifier(n_estimators=500,random_state=42)
rfc.fit(x_train,y_train)


# In[ ]:


preds = rfc.predict(x_test)


# In[ ]:


print('accuracy: ', accuracy_score(y_test,preds))
print(classification_report(y_test,preds))


# # Model Report
# - ## Algorithm Used: Random Forest(estimators=500,random_state=42) 
# ### 1. Preprocessing: Only removing Punctations
# #### a. Bag of words model
# - Accuracy: **77%**
# 
# #### b. Ngrams(1,2) model
# - Accuracy: **76.77%**
# 
# #### c. TF-IDF model
# - Accuracy: **77.98%**
# 
# ### 1) Preprocessing: Removing Stopwords + Punctations
# 
# #### a. Bag of words model
# - Accuracy: **71%**
# 
# #### b. Ngrams(1,2) model
# - Accuracy: **71.2%**

# In[ ]:




