#!/usr/bin/env python
# coding: utf-8

# 

# 

# 

# 

# 

# 

# 

# 

# In[ ]:


import pandas as pd


# 

# In[ ]:


news = pd.read_csv("../input/uci-news-aggregator.csv")


# 

# In[ ]:


print(news.head())


# 

# In[ ]:


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y = encoder.fit_transform(news['CATEGORY'])
print(y[:5])


# In[ ]:


categories = news['CATEGORY']
titles = news['TITLE']
N = len(titles)
print('Number of news',N)


# In[ ]:


labels = list(set(categories))
print('possible categories',labels)


# In[ ]:


for l in labels:
    print('number of ',l,' news',len(news.loc[news['CATEGORY'] == l]))


# 

# In[ ]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
ncategories = encoder.fit_transform(categories)


# 

# 

# In[ ]:


Ntrain = int(N * 0.7)
from sklearn.utils import shuffle
titles, ncategories = shuffle(titles, ncategories, random_state=0)


# In[ ]:


X_train = titles[:Ntrain]
print('X_train.shape',X_train.shape)
y_train = ncategories[:Ntrain]
print('y_train.shape',y_train.shape)
X_test = titles[Ntrain:]
print('X_test.shape',X_test.shape)
y_test = ncategories[Ntrain:]
print('y_test.shape',y_test.shape)


# 

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# In[ ]:


print('Training...')

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
                     ])


# 

# In[ ]:


text_clf = text_clf.fit(X_train, y_train)


# 

# In[ ]:


print('Predicting...')
predicted = text_clf.predict(X_test)


# 

# In[ ]:


from sklearn import metrics

print('accuracy_score',metrics.accuracy_score(y_test,predicted))
print('Reporting...')


# 

# In[ ]:


print(metrics.classification_report(y_test, predicted, target_names=labels))


# 
