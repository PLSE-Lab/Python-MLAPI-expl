#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[ ]:


# imports 
import numpy as np # linear algebra
import pandas as pd # data processing

# visualization
import matplotlib.pyplot as plt


from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS

# model selection
from sklearn.model_selection import train_test_split

# accuracy score
from sklearn.metrics import accuracy_score

# NLP
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Model
from sklearn.naive_bayes import MultinomialNB


# ## Load Dataset

# In[ ]:


# load the data
data = pd.read_csv('../input/nlp-starter-test/social_media_clean_text.csv')


# In[ ]:


# print head of the dataframe
data.head()


# In[ ]:


# print data info
data.info()


# ## Plot WordCloud for each category

# In[ ]:


stopwords = set(STOPWORDS)


# In[ ]:


def wordcloud_plot(name_of_feature):
    plt.figure(figsize=(10, 10))
    wordcloud = WordCloud(
                              background_color='black',
                              stopwords=stopwords,
                              max_words=200,
                              max_font_size=40, 
                              random_state=42
                             ).generate(str(name_of_feature))
    fig = plt.figure(1)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()


# In[ ]:


relevant_text = data[data['choose_one']=='Relevant']['text']
irrelevant_text = data[data['choose_one']=='Not Relevant']['text']


# In[ ]:


wordcloud_plot(relevant_text)


# In[ ]:


wordcloud_plot(irrelevant_text)


# In[ ]:


# Sample dataset
X = data['text']
y = data['choose_one']


# ## Modeling
#    1. CountVEctorizer

# In[ ]:


# Intialize CountVectorizer
cv = CountVectorizer()


# In[ ]:


# fit and transform CountVectorizer 
X1 = cv.fit_transform(X)


# In[ ]:


# Create train and test split with test size 33 %
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.33, random_state=53)


# In[ ]:


# Intiate Naive_bayes
clf = MultinomialNB()


# In[ ]:


clf.fit(X_train, y_train)


# In[ ]:


y_pred_clf = clf.predict(X_test)


# In[ ]:


clf_score = accuracy_score(y_pred_clf, y_test)
print('Accuracy with CountVectorizer : ', clf_score*100)


# 2. TfidfVectorizer

# In[ ]:


# initiate TfidfVEctorizer
tv = TfidfVectorizer()


# In[ ]:


# fit and tranform the tfidfVectorizer
X2 = tv.fit_transform(X)


# In[ ]:


# create train and test split 
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.33, random_state=53)


# In[ ]:


# initialize Naive Bayes
clf = MultinomialNB()


# In[ ]:


clf.fit(X_train, y_train)


# In[ ]:


y_pred_clf = clf.predict(X_test)


# In[ ]:


score = accuracy_score(y_pred_clf, y_test)
print('Accuracy with TfidfVectorizer : ', score*100)


# ## Predict the input

# In[ ]:


text = data['text'].iloc[0]


# In[ ]:


# create Tfidf transform
temp = tv.transform([text])


# In[ ]:


clf.predict(temp)[0]


# Wow, We got the currect prediction.
