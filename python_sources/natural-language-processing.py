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


yelp = pd.read_csv('../input/yelp.csv')


# In[ ]:


yelp.head()


# In[ ]:


yelp.describe()


# In[ ]:


yelp.info()


# In[ ]:


yelp['text'].apply(len)


# In[ ]:


yelp['text length'] = yelp['text'].apply(len)


# In[ ]:


yelp.head()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


FG = sns.FacetGrid(yelp,col='stars')
FG.map(plt.hist,'text length')


# In[ ]:


sns.boxplot(x='stars',y='text length',data=yelp,palette='coolwarm')


# In[ ]:


sns.countplot(x='stars',data=yelp,palette='rainbow')


# In[ ]:


stars = yelp.groupby('stars').mean()
stars


# In[ ]:


stars.corr()


# In[ ]:


sns.heatmap(stars.corr(), cmap= 'coolwarm',annot = True)


# In[ ]:


yelp_stars = yelp[(yelp.stars==1) | (yelp.stars==5)]


# In[ ]:


X = yelp_stars['text']


# In[ ]:


y = yelp_stars['stars']


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()


# In[ ]:


X = cv.fit_transform(X)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# In[ ]:


nb.fit(X_train,y_train)


# In[ ]:


predictions = nb.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report


# In[ ]:


print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))


# In[ ]:


from sklearn.feature_extraction.text import  TfidfTransformer


# In[ ]:


from sklearn.pipeline import Pipeline


# In[ ]:


pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


# In[ ]:


X = yelp_stars['text']
y = yelp_stars['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)


# In[ ]:


pipeline.fit(X_train,y_train)


# In[ ]:


predictions__ = pipeline.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,predictions__))
print(classification_report(y_test,predictions__))


# In[ ]:




