#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


yelp = pd.read_csv('../input/yelp.csv')


# In[ ]:


yelp['text_length'] = yelp['text'].apply(len)


# In[ ]:


yelp.head(2)


# In[ ]:


sns.set_style('white')
s = sns.FacetGrid(yelp,col='stars')
s.map(plt.hist,'text_length',bins=50)


# In[ ]:


Distribution is same in case of all star ratings


# In[ ]:


sns.boxplot(x='stars',y='text_length',data= yelp,palette='rainbow')


# In[ ]:


sns.countplot(x='stars',data= yelp,palette='rainbow')


# In[ ]:


stars = yelp.groupby('stars').mean()
stars


# In[ ]:


sns.heatmap(stars.corr(),annot=True)


# In[ ]:


yelp_class = yelp[(yelp['stars']==1) | (yelp['stars']==5)]
yelp_class.info()


# In[ ]:


X = yelp_class['text']
y = yelp_class['stars']


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train,y_train)


# In[ ]:


predictions = nb.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# Lets see what happens if we include TF-IDF to this process using a pipeline.
# 

# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow',CountVectorizer()),
    ('tfidf',TfidfTransformer()),
    ('classifier',MultinomialNB())
])

X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

pipeline.fit(X_train,y_train)

predictionsTF = pipeline.predict(X_test)

print(classification_report(y_test,predictionsTF))


# >** TF-IDF made things worse !!**
# >
